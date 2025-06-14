# cd 275_Project
# py kitti_tuned_hmm.py
# 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


# cd 275_Project
# py kitti_tuned_hmm_fixed.py
# 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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial import KDTree, distance_matrix
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D not available. PLY files will be saved as numpy arrays.")

# Try to import advanced analysis - MOVED TO TOP FOR BETTER ORGANIZATION
try:
    import hmm_advanced_analysis
    HMM_ADVANCED_AVAILABLE = True
    print("✅ Advanced HMM analysis module loaded successfully")
except ImportError:
    HMM_ADVANCED_AVAILABLE = False
    print("⚠️ hmm_advanced_analysis.py not found. Advanced analysis will be skipped.")

@dataclass
class RoadSegment:
    """Enhanced road segment for KITTI-360 data"""
    segment_id: int
    points: np.ndarray
    center: np.ndarray
    width: float
    length: float
    curvature: float
    connected_segments: List[int]
    segment_type: str = 'road'
    point_density: float = 0.0
    elevation_std: float = 0.0

@dataclass
class TrajectoryPoint:
    """Trajectory point with enhanced features"""
    timestamp: float
    position: np.ndarray
    segment_id: Optional[int] = None
    velocity: Optional[float] = None

class MultiMapKITTILoader:
    """Multi-map KITTI-360 data loader"""
    
    def __init__(self, base_data_path: str):
        self.base_data_path = Path(base_data_path)
        self.map_directories = []
        self.discover_maps()
        
    def discover_maps(self):
        """Discover all map directories in the base path"""
        print(f"🗺️ Discovering maps in: {self.base_data_path}")
        
        # Look for drive directories
        drive_pattern = "*drive_*_sync"
        potential_maps = list(self.base_data_path.glob(drive_pattern))
        
        for map_dir in potential_maps:
            static_dir = map_dir / "static"
            if static_dir.exists():
                # Check if it contains point cloud files
                pc_files = list(static_dir.glob("*.ply")) + list(static_dir.glob("*.bin"))
                if pc_files:
                    self.map_directories.append({
                        'name': map_dir.name,
                        'path': static_dir,
                        'drive_path': map_dir,
                        'file_count': len(pc_files)
                    })
        
        print(f"   ✅ Found {len(self.map_directories)} maps:")
        for i, map_info in enumerate(self.map_directories):
            print(f"      {i+1}. {map_info['name']} ({map_info['file_count']} files)")

class KITTITunedLoader:
    """KITTI-360 data loader tuned for your specific dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.coordinate_bounds = None
        
    def load_point_cloud(self, filename: str) -> np.ndarray:
        """Load and analyze point cloud"""
        file_path = self.data_path / filename
        
        print(f"📂 Loading: {filename}")
        
        if file_path.suffix == '.ply' and OPEN3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
        elif file_path.suffix == '.bin':
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Store coordinate bounds for later use
        self.coordinate_bounds = {
            'x_min': points[:, 0].min(), 'x_max': points[:, 0].max(),
            'y_min': points[:, 1].min(), 'y_max': points[:, 1].max(),
            'z_min': points[:, 2].min(), 'z_max': points[:, 2].max()
        }
        
        print(f"✅ Loaded {len(points):,} points")
        print(f"   Bounds: X[{self.coordinate_bounds['x_min']:.1f}, {self.coordinate_bounds['x_max']:.1f}], "
              f"Y[{self.coordinate_bounds['y_min']:.1f}, {self.coordinate_bounds['y_max']:.1f}], "
              f"Z[{self.coordinate_bounds['z_min']:.1f}, {self.coordinate_bounds['z_max']:.1f}]")
        
        return points
    
    def discover_files(self):
        """Discover available files"""
        pc_files = list(self.data_path.glob("*.ply")) + list(self.data_path.glob("*.bin"))
        traj_files = list(self.data_path.glob("*poses*.txt")) + list(self.data_path.glob("*trajectory*.txt"))
        return pc_files, traj_files

class KITTIRoadExtractor:
    """Road network extractor tuned for KITTI-360 characteristics"""
    
    def __init__(self, grid_size: float = 8.0, height_percentile: float = 15.0, min_density: int = 100):
        self.grid_size = grid_size
        self.height_percentile = height_percentile
        self.min_density = min_density
        
    def extract_road_surface(self, points: np.ndarray) -> np.ndarray:
        """Enhanced road surface extraction for KITTI-360"""
        
        print(f"🛤️ Extracting road surface...")
        
        # Use percentile-based height filtering (better for KITTI-360)
        z_threshold = np.percentile(points[:, 2], self.height_percentile)
        height_range = 2.0  # Allow 2m height variation
        
        road_mask = (points[:, 2] >= z_threshold - 0.5) & (points[:, 2] <= z_threshold + height_range)
        road_points = points[road_mask]
        
        print(f"   📊 Height threshold: {z_threshold:.2f}m")
        print(f"   🛣️ Road points: {len(road_points):,} ({len(road_points)/len(points)*100:.1f}% of total)")
        
        return road_points
    
    def segment_road_network(self, road_points: np.ndarray) -> Dict[int, RoadSegment]:
        """Enhanced road segmentation with better connectivity"""
        
        print(f"🔨 Segmenting road network (grid size: {self.grid_size}m)...")
        
        # Create adaptive grid based on point density
        x_min, y_min = road_points[:, :2].min(axis=0)
        x_max, y_max = road_points[:, :2].max(axis=0)
        
        x_bins = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_bins = np.arange(y_min, y_max + self.grid_size, self.grid_size)
        
        segments = {}
        segment_id = 0
        
        print(f"   📐 Grid: {len(x_bins)}x{len(y_bins)} cells")
        
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                # Get points in this grid cell
                mask = ((road_points[:, 0] >= x_bins[i]) & 
                       (road_points[:, 0] < x_bins[i+1]) &
                       (road_points[:, 1] >= y_bins[j]) & 
                       (road_points[:, 1] < y_bins[j+1]))
                
                cell_points = road_points[mask]
                
                if len(cell_points) >= self.min_density:
                    # Enhanced feature calculation
                    center = cell_points.mean(axis=0)
                    width = self._estimate_width_improved(cell_points)
                    length = self.grid_size
                    curvature = self._estimate_curvature_improved(cell_points)
                    point_density = len(cell_points) / (self.grid_size ** 2)
                    elevation_std = cell_points[:, 2].std()
                    
                    segment = RoadSegment(
                        segment_id=segment_id,
                        points=cell_points,
                        center=center,
                        width=width,
                        length=length,
                        curvature=curvature,
                        connected_segments=[],
                        point_density=point_density,
                        elevation_std=elevation_std
                    )
                    
                    segments[segment_id] = segment
                    segment_id += 1
        
        print(f"   ✅ Created {len(segments)} road segments")
        
        # Enhanced connectivity determination
        self._determine_enhanced_connectivity(segments)
        
        return segments
    
    def _estimate_width_improved(self, points: np.ndarray) -> float:
        """Improved width estimation using PCA"""
        if len(points) < 10:
            return 4.0  # Default road width
        
        # Use PCA to find principal directions
        centered_points = points[:, :2] - points[:, :2].mean(axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Width is approximately 2.5 * std along minor axis
        minor_axis_std = np.sqrt(eigenvals.min())
        width = 2.5 * minor_axis_std
        
        # Clamp to reasonable road width range
        return np.clip(width, 2.5, 12.0)
    
    def _estimate_curvature_improved(self, points: np.ndarray) -> float:
        """Improved curvature estimation"""
        if len(points) < 15:
            return 0.0
        
        try:
            # Use sliding window to estimate local curvature
            xy_points = points[:, :2]
            center = xy_points.mean(axis=0)
            
            # Find points along principal axis
            centered = xy_points - center
            cov_matrix = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
            
            # Project points onto principal axis
            principal_axis = eigenvecs[:, eigenvals.argmax()]
            projections = centered @ principal_axis
            
            # Sort points along principal axis
            sorted_indices = np.argsort(projections)
            sorted_points = xy_points[sorted_indices]
            
            # Estimate curvature using three-point method
            if len(sorted_points) >= 3:
                curvatures = []
                for i in range(1, len(sorted_points) - 1):
                    p1, p2, p3 = sorted_points[i-1], sorted_points[i], sorted_points[i+1]
                    
                    # Calculate curvature using circumcircle
                    a = np.linalg.norm(p2 - p1)
                    b = np.linalg.norm(p3 - p2)
                    c = np.linalg.norm(p3 - p1)
                    
                    if a > 0 and b > 0 and c > 0:
                        area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                        if area > 1e-6:
                            curvature = 4 * area / (a * b * c)
                            curvatures.append(curvature)
                
                return np.mean(curvatures) if curvatures else 0.0
        except:
            pass
        
        return 0.0
    
    def _determine_enhanced_connectivity(self, segments: Dict[int, RoadSegment]):
        """Enhanced connectivity determination"""
        
        print("🔗 Determining segment connectivity...")
        
        centers = np.array([seg.center[:2] for seg in segments.values()])
        segment_ids = list(segments.keys())
        
        # Use KDTree for efficient neighbor finding
        tree = KDTree(centers)
        
        # Adaptive connectivity radius based on grid size
        connectivity_radius = self.grid_size * 1.6
        
        for i, (seg_id, segment) in enumerate(segments.items()):
            # Find nearby segments
            neighbor_indices = tree.query_ball_point(centers[i], connectivity_radius)
            
            connected = []
            for neighbor_idx in neighbor_indices:
                neighbor_id = segment_ids[neighbor_idx]
                
                if neighbor_id != seg_id:
                    neighbor_segment = segments[neighbor_id]
                    
                    # Additional connectivity criteria
                    distance = np.linalg.norm(centers[i] - centers[neighbor_idx])
                    height_diff = abs(segment.center[2] - neighbor_segment.center[2])
                    
                    # Connect if close and similar elevation
                    if distance <= connectivity_radius and height_diff <= 1.0:
                        connected.append(neighbor_id)
            
            segment.connected_segments = connected
        
        # Report connectivity statistics
        connection_counts = [len(seg.connected_segments) for seg in segments.values()]
        avg_connections = np.mean(connection_counts)
        print(f"   📊 Average connections per segment: {avg_connections:.2f}")

class KITTITunedHMM:
    """HMM specifically tuned for KITTI-360 road networks"""
    
    def __init__(self, n_states: int):
        self.n_states = n_states
        self.initial_probs = np.ones(n_states) / n_states
        self.transition_probs = np.ones((n_states, n_states)) / n_states
        self.emission_means = None
        self.emission_covs = None
        self.state_to_segment = {}
        self.segment_to_state = {}
        self.segments = {}
        
        # FIXED: Add these attributes required by advanced analysis
        self.is_trained = False
        self.convergence_info = {}
        
    def fit_segments_to_states(self, segments: Dict[int, RoadSegment]):
        """Intelligent segment-to-state mapping using clustering"""
        
        print(f"🧠 Mapping {len(segments)} segments to {self.n_states} states...")
        
        self.segments = segments
        segment_ids = list(segments.keys())
        
        if len(segment_ids) <= self.n_states:
            # Direct mapping for small networks
            for i, seg_id in enumerate(segment_ids):
                self.state_to_segment[i] = seg_id
                self.segment_to_state[seg_id] = i
            
            # Fill unused states
            for i in range(len(segment_ids), self.n_states):
                self.state_to_segment[i] = segment_ids[0]
        else:
            # Clustering-based mapping for larger networks
            features = []
            for segment in segments.values():
                feature_vector = [
                    segment.center[0], segment.center[1],  # Position
                    segment.width, segment.curvature,      # Geometry
                    segment.point_density,                 # Density
                    len(segment.connected_segments),       # Connectivity
                    segment.elevation_std                  # Elevation variation
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use K-means clustering
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            for i, seg_id in enumerate(segment_ids):
                cluster = cluster_labels[i]
                self.segment_to_state[seg_id] = cluster
                
                if cluster not in self.state_to_segment:
                    self.state_to_segment[cluster] = seg_id
        
        used_states = len(set(self.segment_to_state.values()))
        print(f"   ✅ Mapping complete. Using {used_states}/{self.n_states} states")
    
    def create_realistic_trajectories(self, segments: Dict[int, RoadSegment], 
                                    n_trajectories: int = 25) -> Tuple[List[List[int]], List[List[np.ndarray]]]:
        """Create realistic trajectories following road network connectivity"""
        
        print(f"🚗 Creating {n_trajectories} realistic trajectories...")
        
        # Build road network graph
        G = nx.Graph()
        for seg_id, segment in segments.items():
            G.add_node(seg_id, 
                      pos=segment.center[:2],
                      features=[segment.width, segment.curvature, 
                               segment.point_density, len(segment.connected_segments)])
            
            for connected_id in segment.connected_segments:
                if connected_id in segments:
                    G.add_edge(seg_id, connected_id)
        
        print(f"   🌐 Network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Find connected components
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len) if components else set()
        
        print(f"   🔗 Largest connected component: {len(largest_component)} segments")
        
        state_sequences = []
        observation_sequences = []
        
        for traj_id in range(n_trajectories):
            try:
                # Pick start and end from largest component
                if len(largest_component) < 5:
                    continue
                
                available_nodes = list(largest_component)
                start_seg = np.random.choice(available_nodes)
                
                # Find reachable end points (not too close, not too far)
                start_pos = segments[start_seg].center[:2]
                valid_ends = []
                
                for node in available_nodes:
                    if node != start_seg:
                        end_pos = segments[node].center[:2]
                        distance = np.linalg.norm(end_pos - start_pos)
                        if 30 < distance < 120:  # Reasonable trajectory length
                            valid_ends.append(node)
                
                if not valid_ends:
                    continue
                
                end_seg = np.random.choice(valid_ends)
                
                # Find path using shortest path
                try:
                    path = nx.shortest_path(G, start_seg, end_seg)
                    
                    # Add some realistic deviations
                    if len(path) > 4 and np.random.random() < 0.3:
                        path = self._add_realistic_detour(G, path, segments)
                    
                    # Convert to states and observations
                    states = []
                    observations = []
                    
                    for seg_id in path:
                        if seg_id in self.segment_to_state:
                            state = self.segment_to_state[seg_id]
                            states.append(state)
                            
                            segment = segments[seg_id]
                            obs = np.array([
                                segment.width,
                                segment.curvature,
                                segment.point_density / 100.0,  # Normalize
                                len(segment.connected_segments) / 8.0,  # Normalize
                                segment.elevation_std,
                                np.random.normal(0, 0.1)  # Small noise
                            ])
                            observations.append(obs)
                    
                    if len(states) >= 3:  # Minimum trajectory length
                        state_sequences.append(states)
                        observation_sequences.append(observations)
                        
                except nx.NetworkXNoPath:
                    continue
                    
            except Exception as e:
                continue
        
        print(f"   ✅ Generated {len(state_sequences)} valid trajectories")
        
        if len(state_sequences) == 0:
            print("   ⚠️ No valid trajectories generated, creating fallback trajectories...")
            return self._create_fallback_trajectories(segments)
        
        return state_sequences, observation_sequences
    
    def _add_realistic_detour(self, G: nx.Graph, path: List[int], segments: Dict) -> List[int]:
        """Add realistic detour to a path"""
        if len(path) < 4:
            return path
        
        # Pick a middle point to deviate from
        detour_start_idx = len(path) // 3
        detour_end_idx = 2 * len(path) // 3
        
        detour_start = path[detour_start_idx]
        detour_end = path[detour_end_idx]
        
        # Find alternative path through a neighbor
        neighbors = list(G.neighbors(detour_start))
        for neighbor in neighbors:
            if neighbor not in path[:detour_end_idx + 1]:
                try:
                    detour_path = nx.shortest_path(G, neighbor, detour_end)
                    if len(detour_path) <= 4:  # Not too long
                        new_path = path[:detour_start_idx + 1] + detour_path + path[detour_end_idx + 1:]
                        return new_path
                except:
                    continue
        
        return path
    
    def _create_fallback_trajectories(self, segments: Dict) -> Tuple[List[List[int]], List[List[np.ndarray]]]:
        """Create fallback trajectories when network-based generation fails"""
        
        print("   🔧 Creating fallback trajectories...")
        
        segment_ids = list(segments.keys())
        state_sequences = []
        observation_sequences = []
        
        for _ in range(10):
            # Create random walk with connectivity bias
            current_seg = np.random.choice(segment_ids)
            path = [current_seg]
            
            for _ in range(15):  # Max length
                segment = segments[current_seg]
                
                # Prefer connected segments, but allow random jumps
                if segment.connected_segments and np.random.random() < 0.7:
                    next_seg = np.random.choice(segment.connected_segments)
                else:
                    next_seg = np.random.choice(segment_ids)
                
                path.append(next_seg)
                current_seg = next_seg
            
            # Convert to states and observations
            states = []
            observations = []
            
            for seg_id in path:
                if seg_id in self.segment_to_state:
                    state = self.segment_to_state[seg_id]
                    states.append(state)
                    
                    segment = segments[seg_id]
                    obs = np.array([
                        segment.width, segment.curvature,
                        segment.point_density / 100.0,
                        len(segment.connected_segments) / 8.0,
                        segment.elevation_std,
                        np.random.normal(0, 0.1)
                    ])
                    observations.append(obs)
            
            if len(states) >= 3:
                state_sequences.append(states)
                observation_sequences.append(observations)
        
        return state_sequences, observation_sequences
    
    def train_baum_welch(self, state_sequences: List[List[int]], 
                        observation_sequences: List[List[np.ndarray]], 
                        max_iterations: int = 20):
        """Enhanced Baum-Welch training"""
        
        if not state_sequences:
            print("❌ No training data available!")
            return
        
        n_obs_features = len(observation_sequences[0][0])
        print(f"🔧 Training HMM: {len(state_sequences)} sequences, {n_obs_features} features")
        
        # Improved initialization
        all_observations = np.vstack([np.array(obs_seq) for obs_seq in observation_sequences])
        
        # Initialize means using k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        kmeans.fit(all_observations)
        self.emission_means = kmeans.cluster_centers_
        
        # Initialize covariances
        self.emission_covs = np.array([np.cov(all_observations.T) * 0.5 for _ in range(self.n_states)])
        
        # Initialize transitions with connectivity bias
        self._initialize_transitions_with_connectivity()
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            print(f"   🔄 Iteration {iteration + 1}/{max_iterations}")
            
            total_log_likelihood = 0
            gamma_sum = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            emission_numerator = np.zeros((self.n_states, n_obs_features))
            emission_cov_numerator = np.zeros((self.n_states, n_obs_features, n_obs_features))
            emission_denominator = np.zeros(self.n_states)
            
            valid_sequences = 0
            
            for observations in observation_sequences:
                try:
                    alpha, beta, log_likelihood = self._forward_backward_stable(observations)
                    total_log_likelihood += log_likelihood
                    valid_sequences += 1
                    
                    T = len(observations)
                    gamma = alpha * beta
                    gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)
                    
                    gamma_sum += gamma.sum(axis=0)
                    
                    # Update emission statistics
                    for t in range(T):
                        for i in range(self.n_states):
                            weight = gamma[t, i]
                            emission_numerator[i] += weight * observations[t]
                            emission_denominator[i] += weight
                            
                            diff = observations[t] - self.emission_means[i]
                            emission_cov_numerator[i] += weight * np.outer(diff, diff)
                    
                    # Update transition statistics
                    for t in range(T-1):
                        xi_t = np.zeros((self.n_states, self.n_states))
                        for i in range(self.n_states):
                            for j in range(self.n_states):
                                xi_t[i, j] = (alpha[t, i] * self.transition_probs[i, j] * 
                                            self._emission_prob_stable(observations[t+1], j) * 
                                            beta[t+1, j])
                        
                        xi_t = xi_t / (xi_t.sum() + 1e-10)
                        xi_sum += xi_t
                        
                except Exception as e:
                    continue
            
            # M-step
            self.transition_probs = xi_sum / (gamma_sum[:, np.newaxis] + 1e-10)
            self.transition_probs = self.transition_probs / (self.transition_probs.sum(axis=1, keepdims=True) + 1e-10)
            
            for i in range(self.n_states):
                if emission_denominator[i] > 1e-10:
                    self.emission_means[i] = emission_numerator[i] / emission_denominator[i]
                    self.emission_covs[i] = emission_cov_numerator[i] / emission_denominator[i] + np.eye(n_obs_features) * 0.01
            
            improvement = total_log_likelihood - prev_log_likelihood
            print(f"      Log likelihood: {total_log_likelihood:.2f} (Δ: {improvement:.4f})")
            
            if abs(improvement) < 0.01:
                print(f"   ✅ Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = total_log_likelihood
        
        # FIXED: Mark as trained and store convergence info
        self.is_trained = True
        self.convergence_info = {
            'final_log_likelihood': total_log_likelihood,
            'iterations': iteration + 1,
            'valid_sequences': valid_sequences
        }
        
        print(f"   ✅ Training complete with {valid_sequences} valid sequences")
    
    def _initialize_transitions_with_connectivity(self):
        """Initialize transition matrix using road connectivity"""
        self.transition_probs = np.ones((self.n_states, self.n_states)) * 0.001
        
        for i in range(self.n_states):
            if i in self.state_to_segment:
                seg_id = self.state_to_segment[i]
                if seg_id in self.segments:
                    segment = self.segments[seg_id]
                    
                    # Add higher probability for connected segments
                    for connected_id in segment.connected_segments:
                        if connected_id in self.segment_to_state:
                            j = self.segment_to_state[connected_id]
                            self.transition_probs[i, j] += 0.1
                    
                    # Add self-transition probability
                    self.transition_probs[i, i] += 0.05
        
        # Normalize
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=1, keepdims=True)
    
    def _forward_backward_stable(self, observations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Numerically stable forward-backward algorithm"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        
        # Forward pass with scaling
        alpha[0] = self.initial_probs * self._emission_prob_stable(observations[0], np.arange(self.n_states))
        alpha[0] = alpha[0] / (alpha[0].sum() + 1e-10)
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * self._emission_prob_stable(observations[t], j)
            alpha[t] = alpha[t] / (alpha[t].sum() + 1e-10)
        
        # Backward pass
        beta[T-1] = 1
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_probs[i] * 
                                  self._emission_prob_stable(observations[t+1], np.arange(self.n_states)) * 
                                  beta[t+1])
            beta[t] = beta[t] / (beta[t].sum() + 1e-10)
        
        log_likelihood = np.log(alpha[T-1].sum() + 1e-10)
        return alpha, beta, log_likelihood
    
    def _emission_prob_stable(self, observation: np.ndarray, state) -> float:
        """Stable emission probability calculation"""
        if isinstance(state, int):
            diff = observation - self.emission_means[state]
            try:
                cov_inv = np.linalg.pinv(self.emission_covs[state])
                return np.exp(-0.5 * diff.T @ cov_inv @ diff) + 1e-10
            except:
                return 1e-10
        else:
            probs = np.zeros(len(state))
            for i, s in enumerate(state):
                diff = observation - self.emission_means[s]
                try:
                    cov_inv = np.linalg.pinv(self.emission_covs[s])
                    probs[i] = np.exp(-0.5 * diff.T @ cov_inv @ diff) + 1e-10
                except:
                    probs[i] = 1e-10
            return probs
    
    def viterbi_decode(self, start_segment: int, end_segment: int, max_length: int = 25) -> List[int]:
        """Enhanced Viterbi decoding"""
        
        print(f"🎯 Generating route: {start_segment} → {end_segment}")
        
        if start_segment not in self.segment_to_state or end_segment not in self.segment_to_state:
            print("   ❌ Start or end segment not mapped to states")
            return []
        
        start_state = self.segment_to_state[start_segment]
        end_state = self.segment_to_state[end_segment]
        
        # Use log probabilities
        log_prob = np.full((max_length, self.n_states), -np.inf)
        path = np.zeros((max_length, self.n_states), dtype=int)
        
        log_prob[0, start_state] = 0
        
        for t in range(1, max_length):
            for j in range(self.n_states):
                for i in range(self.n_states):
                    if log_prob[t-1, i] > -np.inf:
                        transition_prob = self.transition_probs[i, j]
                        if transition_prob > 0:
                            score = log_prob[t-1, i] + np.log(transition_prob)
                            if score > log_prob[t, j]:
                                log_prob[t, j] = score
                                path[t, j] = i
        
        # Find best path to end state
        best_t = -1
        best_score = -np.inf
        
        for t in range(max_length):
            if log_prob[t, end_state] > best_score:
                best_score = log_prob[t, end_state]
                best_t = t
        
        if best_t == -1:
            print("   ❌ No path found")
            return []
        
        # Backtrack
        route_states = [end_state]
        current_state = end_state
        
        for t in range(best_t, 0, -1):
            current_state = path[t, current_state]
            route_states.append(current_state)
        
        route_states.reverse()
        
        # Convert to segments
        route_segments = [self.state_to_segment[state] for state in route_states 
                         if state in self.state_to_segment]
        
        print(f"   ✅ Generated route with {len(route_segments)} segments")
        return route_segments

def process_single_point_cloud(loader: KITTITunedLoader, pc_file: Path, file_index: int, total_files: int):
    """Process a single point cloud file"""
    
    print(f"\n{'='*80}")
    print(f"🔍 PROCESSING FILE {file_index + 1}/{total_files}: {pc_file.name}")
    print(f"{'='*80}")
    
    try:
        # Load point cloud
        points = loader.load_point_cloud(pc_file.name)
        
        # Extract road network
        print(f"\n🛤️ Extracting road network...")
        extractor = KITTIRoadExtractor(grid_size=6.0, min_density=80)
        road_points = extractor.extract_road_surface(points)
        segments = extractor.segment_road_network(road_points)
        
        if not segments:
            print("❌ No road segments created!")
            return None
        
        # Setup and train HMM
        print(f"\n🧠 Setting up HMM...")
        n_states = min(25, max(10, len(segments) // 5))
        hmm = KITTITunedHMM(n_states)
        hmm.fit_segments_to_states(segments)
        
        print(f"\n🚗 Creating training data...")
        state_sequences, observation_sequences = hmm.create_realistic_trajectories(segments)
        
        if not state_sequences:
            print("❌ No training sequences generated!")
            return None
        
        print(f"\n🔧 Training HMM...")
        hmm.train_baum_welch(state_sequences, observation_sequences)
        
        # Generate routes
        print(f"\n🎯 Generating routes...")
        segment_ids = list(segments.keys())
        successful_routes = []
        
        for attempt in range(3):  # Fewer routes per file for speed
            start_idx = attempt * len(segment_ids) // 6
            end_idx = -(attempt + 1) * len(segment_ids) // 6
            
            start_seg = segment_ids[start_idx]
            end_seg = segment_ids[end_idx]
            
            route = hmm.viterbi_decode(start_seg, end_seg)
            
            if route and len(route) > 2:
                successful_routes.append((start_seg, end_seg, route))
                print(f"   ✅ Route {attempt + 1}: {len(route)} segments")
        
        print(f"\n✅ File {file_index + 1} completed: {len(successful_routes)} successful routes!")
        
        # Return comprehensive results
        return {
            'file_index': file_index,
            'filename': pc_file.name,
            'points': points,
            'road_points': road_points,
            'segments': segments,
            'hmm': hmm,
            'successful_routes': successful_routes,
            'coordinate_bounds': loader.coordinate_bounds.copy()
        }
        
    except Exception as e:
        print(f"❌ Error processing {pc_file.name}: {e}")
        return None

def save_ply_results(segments: Dict[int, RoadSegment], output_path: str, map_name: str):
    """Save road segments as PLY files"""
    
    print(f"💾 Saving PLY results for {map_name}...")
    
    output_dir = Path(output_path) / "ply_results" / map_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if OPEN3D_AVAILABLE:
        # Save all road points as one PLY
        all_road_points = []
        segment_colors = []
        
        # Generate colors for each segment
        colors = plt.cm.Set1(np.linspace(0, 1, len(segments)))
        
        for i, (seg_id, segment) in enumerate(segments.items()):
            points = segment.points
            all_road_points.append(points)
            
            # Assign color to this segment's points
            color = colors[i % len(colors)][:3]  # RGB only
            segment_colors.extend([color] * len(points))
        
        if all_road_points:
            all_points = np.vstack(all_road_points)
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(np.array(segment_colors))
            
            # Save main PLY file
            main_ply_path = output_dir / f"{map_name}_road_network.ply"
            o3d.io.write_point_cloud(str(main_ply_path), pcd)
            print(f"   ✅ Saved: {main_ply_path}")
            
            # Save individual segment PLY files (limited to avoid too many files)
            max_individual_segments = 50
            segment_list = list(segments.items())
            if len(segment_list) > max_individual_segments:
                segment_list = segment_list[:max_individual_segments]
                print(f"   📝 Saving first {max_individual_segments} individual segments...")
            
            for seg_id, segment in segment_list:
                seg_pcd = o3d.geometry.PointCloud()
                seg_pcd.points = o3d.utility.Vector3dVector(segment.points)
                
                # Single color for this segment
                color = colors[seg_id % len(colors)][:3]
                seg_colors = np.tile(color, (len(segment.points), 1))
                seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)
                
                seg_ply_path = output_dir / f"segment_{seg_id:04d}.ply"
                o3d.io.write_point_cloud(str(seg_ply_path), seg_pcd)
            
            print(f"   ✅ Saved {len(segment_list)} individual segment PLY files")
    else:
        print("   ⚠️ Open3D not available, saving as numpy arrays...")
        
        # Save as numpy arrays
        for seg_id, segment in segments.items():
            seg_file = output_dir / f"segment_{seg_id:04d}.npy"
            np.save(seg_file, segment.points)
        
        print(f"   ✅ Saved {len(segments)} segment numpy files")

def create_complete_trajectory_analysis(all_results: List[Dict], map_name: str):
    """Create comprehensive analysis for a single map (like original code)"""
    
    print(f"\n🗺️ Creating Complete Trajectory Analysis for {map_name}")
    print("=" * 60)
    
    # 1. Summary statistics
    print("\n1. 📊 Summary Statistics:")
    total_segments = sum(len(result['segments']) for result in all_results)
    total_routes = sum(len(result['successful_routes']) for result in all_results)
    
    print(f"   📁 Files processed: {len(all_results)}")
    print(f"   🛣️ Total road segments: {total_segments:,}")
    print(f"   🎯 Total routes generated: {total_routes}")
    print(f"   📊 Average segments per file: {total_segments / len(all_results):.1f}")
    
    # File-by-file breakdown
    print(f"\n   📋 File breakdown:")
    for result in all_results:
        bounds = result['coordinate_bounds']
        print(f"      {result['filename']}: {len(result['segments'])} segments, "
              f"{len(result['successful_routes'])} routes, "
              f"bounds: X[{bounds['x_min']:.0f}-{bounds['x_max']:.0f}], "
              f"Y[{bounds['y_min']:.0f}-{bounds['y_max']:.0f}]")
    
    # 2. Coordinate analysis
    print(f"\n2. 📍 Spatial Coverage Analysis:")
    all_x_coords = []
    all_y_coords = []
    
    for result in all_results:
        bounds = result['coordinate_bounds']
        all_x_coords.extend([bounds['x_min'], bounds['x_max']])
        all_y_coords.extend([bounds['y_min'], bounds['y_max']])
    
    global_x_min, global_x_max = min(all_x_coords), max(all_x_coords)
    global_y_min, global_y_max = min(all_y_coords), max(all_y_coords)
    
    print(f"   🌍 Global coverage:")
    print(f"      X range: [{global_x_min:.0f}, {global_x_max:.0f}] = {global_x_max - global_x_min:.0f}m")
    print(f"      Y range: [{global_y_min:.0f}, {global_y_max:.0f}] = {global_y_max - global_y_min:.0f}m")
    print(f"      Total area: {(global_x_max - global_x_min) * (global_y_max - global_y_min) / 1000000:.2f} km²")
    
    # 3. Trajectory progression analysis
    print(f"\n3. 🚗 Trajectory Progression Analysis:")
    trajectory_centers = []
    
    for result in all_results:
        bounds = result['coordinate_bounds']
        center_x = (bounds['x_min'] + bounds['x_max']) / 2
        center_y = (bounds['y_min'] + bounds['y_max']) / 2
        trajectory_centers.append([center_x, center_y])
    
    trajectory_centers = np.array(trajectory_centers)
    
    # Calculate total trajectory distance
    total_distance = 0
    for i in range(1, len(trajectory_centers)):
        distance = np.linalg.norm(trajectory_centers[i] - trajectory_centers[i-1])
        total_distance += distance
    
    print(f"   📏 Vehicle trajectory:")
    print(f"      Total distance: {total_distance:.1f} meters")
    print(f"      Average step: {total_distance / (len(trajectory_centers) - 1):.1f} meters")
    print(f"      Start position: ({trajectory_centers[0, 0]:.1f}, {trajectory_centers[0, 1]:.1f})")
    print(f"      End position: ({trajectory_centers[-1, 0]:.1f}, {trajectory_centers[-1, 1]:.1f})")
    
    # 4. Create visualizations
    print(f"\n4. 📈 Creating trajectory visualizations...")
    create_trajectory_visualizations(all_results, trajectory_centers, map_name)
    
    # 5. Route connectivity analysis
    print(f"\n5. 🔗 Route Connectivity Analysis:")
    analyze_route_connectivity(all_results)

def create_trajectory_visualizations(all_results: List[Dict], trajectory_centers: np.ndarray, map_name: str):
    """Create comprehensive visualizations of the complete trajectory"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Complete Trajectory Analysis - {map_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Complete trajectory overview
    ax = axes[0, 0]
    
    # Plot vehicle trajectory
    ax.plot(trajectory_centers[:, 0], trajectory_centers[:, 1], 'red', linewidth=4, 
           marker='o', markersize=8, label='Vehicle Trajectory')
    ax.scatter(trajectory_centers[0, 0], trajectory_centers[0, 1], 
              c='green', s=200, marker='s', label='Start', zorder=5)
    ax.scatter(trajectory_centers[-1, 0], trajectory_centers[-1, 1], 
              c='red', s=200, marker='s', label='End', zorder=5)
    
    # Add file indices
    for i, center in enumerate(trajectory_centers):
        ax.annotate(f'{i+1}', (center[0], center[1]), xytext=(10, 10), 
                   textcoords='offset points', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_title('Complete Vehicle Trajectory\n(KITTI-360 Sequence)', fontweight='bold')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Road segment coverage over time
    ax = axes[0, 1]
    
    file_indices = [result['file_index'] + 1 for result in all_results]
    segment_counts = [len(result['segments']) for result in all_results]
    route_counts = [len(result['successful_routes']) for result in all_results]
    
    ax.bar(file_indices, segment_counts, alpha=0.7, label='Road Segments', color='blue')
    ax2 = ax.twinx()
    ax2.bar([i + 0.3 for i in file_indices], route_counts, alpha=0.7, label='Generated Routes', 
           width=0.3, color='orange')
    
    ax.set_xlabel('File Index')
    ax.set_ylabel('Number of Road Segments', color='blue')
    ax2.set_ylabel('Number of Routes', color='orange')
    ax.set_title('Road Network Analysis\nAcross Sequence', fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spatial coverage evolution
    ax = axes[0, 2]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    
    for i, result in enumerate(all_results):
        bounds = result['coordinate_bounds']
        
        # Draw bounding box for each file
        rect_x = [bounds['x_min'], bounds['x_max'], bounds['x_max'], bounds['x_min'], bounds['x_min']]
        rect_y = [bounds['y_min'], bounds['y_min'], bounds['y_max'], bounds['y_max'], bounds['y_min']]
        
        ax.plot(rect_x, rect_y, color=colors[i], alpha=0.7, linewidth=2, 
               label=f'File {i+1}' if i < 5 else None)
        
        # Mark center
        center_x = (bounds['x_min'] + bounds['x_max']) / 2
        center_y = (bounds['y_min'] + bounds['y_max']) / 2
        ax.scatter(center_x, center_y, color=colors[i], s=50, alpha=0.8)
    
    ax.set_title('Spatial Coverage Evolution', fontweight='bold')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    if len(all_results) <= 5:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Road network density heatmap
    ax = axes[1, 0]
    
    # Create grid for density calculation
    all_segment_centers = []
    for result in all_results:
        for segment in result['segments'].values():
            all_segment_centers.append(segment.center[:2])
    
    if all_segment_centers:
        all_segment_centers = np.array(all_segment_centers)
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            all_segment_centers[:, 0], all_segment_centers[:, 1], bins=30
        )
        
        im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        
        plt.colorbar(im, ax=ax, label='Road Segment Density')
    
    ax.set_title('Road Network Density\n(All Files Combined)', fontweight='bold')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    
    # Plot 5: Route quality metrics over time
    ax = axes[1, 1]
    
    avg_route_lengths = []
    for result in all_results:
        if result['successful_routes']:
            lengths = [len(route[2]) for route in result['successful_routes']]
            avg_route_lengths.append(np.mean(lengths))
        else:
            avg_route_lengths.append(0)
    
    ax.plot(file_indices, avg_route_lengths, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Average Route Length (segments)')
    ax.set_title('Route Quality Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Network connectivity analysis
    ax = axes[1, 2]
    
    connectivity_scores = []
    for result in all_results:
        if result['segments']:
            connections = [len(seg.connected_segments) for seg in result['segments'].values()]
            connectivity_scores.append(np.mean(connections))
        else:
            connectivity_scores.append(0)
    
    ax.bar(file_indices, connectivity_scores, alpha=0.7, color='green')
    ax.set_xlabel('File Index')
    ax.set_ylabel('Average Segment Connectivity')
    ax.set_title('Network Connectivity\nAcross Sequence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_route_connectivity(all_results: List[Dict]):
    """Analyze connectivity between routes across different files"""
    
    print("   🔍 Cross-file route analysis:")
    
    # Try to find potential connections between consecutive files
    connections_found = 0
    total_comparisons = 0
    
    for i in range(len(all_results) - 1):
        current_result = all_results[i]
        next_result = all_results[i + 1]
        
        current_bounds = current_result['coordinate_bounds']
        next_bounds = next_result['coordinate_bounds']
        
        # Check for spatial overlap
        x_overlap = not (current_bounds['x_max'] < next_bounds['x_min'] or 
                        current_bounds['x_min'] > next_bounds['x_max'])
        y_overlap = not (current_bounds['y_max'] < next_bounds['y_min'] or 
                        current_bounds['y_min'] > next_bounds['y_max'])
        
        if x_overlap and y_overlap:
            connections_found += 1
            
            # Calculate overlap area
            overlap_x = min(current_bounds['x_max'], next_bounds['x_max']) - max(current_bounds['x_min'], next_bounds['x_min'])
            overlap_y = min(current_bounds['y_max'], next_bounds['y_max']) - max(current_bounds['y_min'], next_bounds['y_min'])
            overlap_area = overlap_x * overlap_y
            
            print(f"      Files {i+1}↔{i+2}: Overlap area {overlap_area:.0f} m²")
        
        total_comparisons += 1
    
    print(f"   📊 Connectivity summary:")
    print(f"      Connected file pairs: {connections_found}/{total_comparisons}")
    if total_comparisons > 0:
        print(f"      Connectivity ratio: {connections_found/total_comparisons*100:.1f}%")
        
        if connections_found > len(all_results) * 0.7:
            print(f"   ✅ High connectivity - good trajectory continuity!")
        elif connections_found > len(all_results) * 0.3:
            print(f"   ⚠️ Moderate connectivity - some gaps in trajectory")
        else:
            print(f"   ❌ Low connectivity - fragmented trajectory")

def process_single_map_comprehensive(map_info: Dict, output_base_path: str) -> Dict:
    """Process a single map with full comprehensive analysis (like original code)"""
    
    map_name = map_info['name']
    data_path = str(map_info['path'])
    
    print(f"\n🚀 {map_name.upper()} COMPLETE TRAJECTORY ANALYSIS")
    print("=" * 80)
    print(f"🎯 Processing ALL point cloud files to build complete trajectory!")
    
    try:
        # 1. Discover all files
        print(f"\n1. 📂 Discovering KITTI-360 data...")
        loader = KITTITunedLoader(data_path)
        pc_files, traj_files = loader.discover_files()
        
        if not pc_files:
            print(f"❌ No point cloud files found in {map_name}")
            return None
        
        print(f"   ✅ Found {len(pc_files)} point cloud files to process")
        
        # Sort files by name to ensure chronological order
        pc_files = sorted(pc_files, key=lambda x: x.name)
        
        # 2. Process each point cloud file
        print(f"\n2. 🔄 Processing all {len(pc_files)} files...")
        all_results = []
        
        for i, pc_file in enumerate(pc_files):
            result = process_single_point_cloud(loader, pc_file, i, len(pc_files))
            if result:
                all_results.append(result)
        
        if not all_results:
            print(f"❌ No files processed successfully for {map_name}!")
            return None
        
        print(f"\n✅ Successfully processed {len(all_results)}/{len(pc_files)} files!")
        
        # 3. Create trajectory analysis
        print(f"\n3. 🗺️ Creating complete trajectory analysis...")
        create_complete_trajectory_analysis(all_results, map_name)
        
        # 4. Save PLY results
        print(f"\n💾 Saving PLY results...")
        # Combine all segments for PLY export
        all_segments = {}
        for result in all_results:
            all_segments.update(result['segments'])
        
        save_ply_results(all_segments, output_base_path, map_name)
        
        # FIXED: Enhanced advanced analysis with better error handling
        print(f"\n4. 🔬 Running detailed analysis on best file...")
        best_result = max(all_results, key=lambda x: len(x['successful_routes']))
        print(f"   📊 Selected file: {best_result['filename']} ({len(best_result['successful_routes'])} routes)")
        
        segments = best_result['segments']
        hmm = best_result['hmm']
        segment_ids = list(segments.keys())
        
        if len(segment_ids) >= 2 and HMM_ADVANCED_AVAILABLE:
            # FIXED: Better segment selection for advanced analysis
            try:
                # Choose segments that are more likely to have good paths
                if len(segment_ids) > 10:
                    start_seg = segment_ids[len(segment_ids)//4]  # Not at the very edge
                    end_seg = segment_ids[3*len(segment_ids)//4]
                else:
                    start_seg = segment_ids[0]
                    end_seg = segment_ids[-1]
                
                print(f"\n🔬 DETAILED ANALYSIS: {map_name}")
                print("=" * 60)
                print(f"   🎯 Analysis route: {start_seg} → {end_seg}")
                
                # FIXED: Add validation before calling advanced analysis
                if hasattr(hmm, 'is_trained') and hmm.is_trained:
                    hmm_advanced_analysis.create_comprehensive_analysis(hmm, segments, start_seg, end_seg)
                else:
                    print("   ⚠️ HMM not properly trained, skipping advanced analysis")
                    
            except Exception as e:
                print(f"   ❌ Advanced analysis failed: {e}")
                print("   💡 This is often due to network connectivity issues or data sparsity")
                
        elif not HMM_ADVANCED_AVAILABLE:
            print(f"\n⚠️ hmm_advanced_analysis.py not found. Skipping detailed analysis.")
            print("💡 Add hmm_advanced_analysis.py to enable comprehensive route analysis!")
        else:
            print(f"\n⚠️ Insufficient segments ({len(segment_ids)}) for advanced analysis")
        
        # Calculate overall statistics
        total_segments = len(all_segments)
        total_routes = sum(len(result['successful_routes']) for result in all_results)
        
        # Calculate coordinate bounds
        all_x_coords = []
        all_y_coords = []
        for result in all_results:
            bounds = result['coordinate_bounds']
            all_x_coords.extend([bounds['x_min'], bounds['x_max']])
            all_y_coords.extend([bounds['y_min'], bounds['y_max']])
        
        coordinate_bounds = {
            'x_min': min(all_x_coords), 'x_max': max(all_x_coords),
            'y_min': min(all_y_coords), 'y_max': max(all_y_coords),
        }
        
        print(f"\n🎉 {map_name.upper()} TRAJECTORY ANALYSIS FINISHED!")
        print("=" * 80)
        
        return {
            'map_name': map_name,
            'total_segments': total_segments,
            'total_routes': total_routes,
            'total_files_processed': len(all_results),
            'coordinate_bounds': coordinate_bounds,
            'file_results': all_results,
            'best_result': best_result
        }
        
    except Exception as e:
        print(f"❌ Error processing map {map_name}: {e}")
        return None

def create_multi_map_summary(all_map_results: List[Dict], output_path: str):
    """Create comprehensive summary across all maps"""
    
    print(f"\n🌍 MULTI-MAP SUMMARY ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    total_maps = len(all_map_results)
    total_segments = sum(result['total_segments'] for result in all_map_results)
    total_routes = sum(result['total_routes'] for result in all_map_results)
    total_files = sum(result['total_files_processed'] for result in all_map_results)
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   🗺️ Maps processed: {total_maps}")
    print(f"   📁 Total files: {total_files}")
    print(f"   🛣️ Total segments: {total_segments:,}")
    print(f"   🎯 Total routes: {total_routes}")
    print(f"   📈 Avg segments per map: {total_segments / total_maps:.1f}")
    print(f"   📈 Avg routes per map: {total_routes / total_maps:.1f}")
    
    # Map-by-map breakdown
    print(f"\n📋 MAP BREAKDOWN:")
    for i, result in enumerate(all_map_results):
        bounds = result['coordinate_bounds']
        print(f"   {i+1}. {result['map_name']}:")
        print(f"      📁 Files: {result['total_files_processed']}")
        print(f"      🛣️ Segments: {result['total_segments']}")
        print(f"      🎯 Routes: {result['total_routes']}")
        print(f"      📍 Bounds: X[{bounds['x_min']:.0f}-{bounds['x_max']:.0f}], "
              f"Y[{bounds['y_min']:.0f}-{bounds['y_max']:.0f}]")
    
    # Create comprehensive visualization
    print(f"\n📈 Creating multi-map visualization...")
    create_multi_map_visualization(all_map_results)
    
    # Save summary report
    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'total_maps': total_maps,
        'total_files': total_files,
        'total_segments': total_segments,
        'total_routes': total_routes,
        'maps': []
    }
    
    for result in all_map_results:
        map_summary = {
            'name': result['map_name'],
            'files_processed': result['total_files_processed'],
            'segments': result['total_segments'],
            'routes': result['total_routes'],
            'coordinate_bounds': result['coordinate_bounds']
        }
        summary_report['maps'].append(map_summary)
    
    # Save JSON report
    report_path = Path(output_path) / "multi_map_summary.json"
    with open(report_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"💾 Summary report saved: {report_path}")

def create_multi_map_visualization(all_map_results: List[Dict]):
    """Create comprehensive visualization across all maps"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Map KITTI-360 Analysis Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Maps overview
    ax = axes[0, 0]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_map_results)))
    
    for i, result in enumerate(all_map_results):
        bounds = result['coordinate_bounds']
        
        # Draw bounding box
        rect_x = [bounds['x_min'], bounds['x_max'], bounds['x_max'], bounds['x_min'], bounds['x_min']]
        rect_y = [bounds['y_min'], bounds['y_min'], bounds['y_max'], bounds['y_max'], bounds['y_min']]
        
        ax.plot(rect_x, rect_y, color=colors[i], linewidth=2, alpha=0.7, 
               label=result['map_name'][:20])  # Truncate long names
        
        # Mark center
        center_x = (bounds['x_min'] + bounds['x_max']) / 2
        center_y = (bounds['y_min'] + bounds['y_max']) / 2
        ax.scatter(center_x, center_y, color=colors[i], s=100, alpha=0.8)
        ax.annotate(f"{i+1}", (center_x, center_y), xytext=(10, 10), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_title('All Maps Spatial Coverage', fontweight='bold')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    if len(all_map_results) <= 6:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Segments per map
    ax = axes[0, 1]
    
    map_names = [result['map_name'][:15] for result in all_map_results]  # Truncate names
    segment_counts = [result['total_segments'] for result in all_map_results]
    
    bars = ax.bar(range(len(map_names)), segment_counts, color=colors, alpha=0.7)
    ax.set_xlabel('Map Index')
    ax.set_ylabel('Number of Segments')
    ax.set_title('Road Segments per Map', fontweight='bold')
    ax.set_xticks(range(len(map_names)))
    ax.set_xticklabels([f"{i+1}" for i in range(len(map_names))])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, segment_counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 3: Routes per map
    ax = axes[0, 2]
    
    route_counts = [result['total_routes'] for result in all_map_results]
    
    bars = ax.bar(range(len(map_names)), route_counts, color=colors, alpha=0.7)
    ax.set_xlabel('Map Index')
    ax.set_ylabel('Number of Routes')
    ax.set_title('Generated Routes per Map', fontweight='bold')
    ax.set_xticks(range(len(map_names)))
    ax.set_xticklabels([f"{i+1}" for i in range(len(map_names))])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, route_counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 4: Map sizes (area coverage)
    ax = axes[1, 0]
    
    map_areas = []
    for result in all_map_results:
        bounds = result['coordinate_bounds']
        area = (bounds['x_max'] - bounds['x_min']) * (bounds['y_max'] - bounds['y_min']) / 1000000  # km²
        map_areas.append(area)
    
    bars = ax.bar(range(len(map_names)), map_areas, color=colors, alpha=0.7)
    ax.set_xlabel('Map Index')
    ax.set_ylabel('Area Coverage (km²)')
    ax.set_title('Spatial Coverage per Map', fontweight='bold')
    ax.set_xticks(range(len(map_names)))
    ax.set_xticklabels([f"{i+1}" for i in range(len(map_names))])
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Files processed per map
    ax = axes[1, 1]
    
    files_processed = [result['total_files_processed'] for result in all_map_results]
    
    bars = ax.bar(range(len(map_names)), files_processed, color=colors, alpha=0.7)
    ax.set_xlabel('Map Index')
    ax.set_ylabel('Files Processed')
    ax.set_title('Files Processed per Map', fontweight='bold')
    ax.set_xticks(range(len(map_names)))
    ax.set_xticklabels([f"{i+1}" for i in range(len(map_names))])
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency metrics
    ax = axes[1, 2]
    
    # Calculate segments per file ratio
    efficiency = [result['total_segments'] / max(result['total_files_processed'], 1) 
                 for result in all_map_results]
    
    bars = ax.bar(range(len(map_names)), efficiency, color=colors, alpha=0.7)
    ax.set_xlabel('Map Index')
    ax.set_ylabel('Segments per File')
    ax.set_title('Processing Efficiency\n(Segments/File)', fontweight='bold')
    ax.set_xticks(range(len(map_names)))
    ax.set_xticklabels([f"{i+1}" for i in range(len(map_names))])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def process_all_kitti_maps(base_data_path: str):
    """Process all KITTI-360 maps with comprehensive analysis like original code"""
    
    print("🚀 MULTI-MAP KITTI-360 COMPREHENSIVE TRAJECTORY ANALYSIS")
    print("=" * 100)
    print(f"🎯 Processing ALL maps with FULL analysis like single-map version")
    
    # 1. Discover all maps
    print("\n1. 🗺️ Discovering maps...")
    multi_loader = MultiMapKITTILoader(base_data_path)
    
    if not multi_loader.map_directories:
        print("❌ No maps found!")
        return
    
    # 2. Create output directory
    output_path = Path(base_data_path) / "multi_map_results"
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # 3. Process each map with FULL comprehensive analysis
    print(f"\n2. 🔄 Processing {len(multi_loader.map_directories)} maps with FULL analysis...")
    all_map_results = []
    
    for i, map_info in enumerate(multi_loader.map_directories):
        print(f"\n{'='*50} MAP {i+1}/{len(multi_loader.map_directories)} {'='*50}")
        
        result = process_single_map_comprehensive(map_info, str(output_path))
        if result:
            all_map_results.append(result)
    
    if not all_map_results:
        print("❌ No maps processed successfully!")
        return
    
    print(f"\n3. 📊 Creating multi-map summary...")
    create_multi_map_summary(all_map_results, str(output_path))
    
    print(f"\n🎉 MULTI-MAP COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"✅ Successfully processed {len(all_map_results)}/{len(multi_loader.map_directories)} maps")
    print(f"📁 Results saved in: {output_path}")
    print(f"📊 Each map received FULL analysis like single-map version")
    print(f"📈 PLY files, visualizations, HMM analysis, and summary available")

if __name__ == "__main__":
    # Your multi-map KITTI-360 data path
    base_data_path = r"C:\Users\caleb\OneDrive\Desktop\data_3d_semantics\data_3d_semantics\train"
    
    process_all_kitti_maps(base_data_path)