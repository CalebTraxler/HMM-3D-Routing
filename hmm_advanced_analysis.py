# cd 275_Project
# py hmm_advanced_analysis.py

# cd 275_Project
# py hmm_advanced_analysis_fixed.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
import warnings
import time
import sys
warnings.filterwarnings('ignore')

@dataclass
class RouteMetrics:
    """Comprehensive route evaluation metrics"""
    route_id: str
    segments: List[int]
    total_length: float
    efficiency_ratio: float  # vs shortest path
    curvature_score: float
    width_variance: float
    connectivity_score: float
    diversity_score: float

class AdvancedRouteGenerator:
    """Advanced route generation with multiple algorithms"""
    
    def __init__(self, hmm, segments):
        self.hmm = hmm
        self.segments = segments
        self.road_graph = self._build_road_graph()
        self._validate_hmm_interface()
        
    def _validate_hmm_interface(self):
        """Validate that HMM has required attributes for analysis"""
        required_attrs = ['n_states', 'transition_probs', 'state_to_segment', 'segment_to_state']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(self.hmm, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"⚠️ HMM missing attributes: {missing_attrs}")
            print("   Advanced analysis may be limited")
        
        # Check if HMM is properly trained
        if hasattr(self.hmm, 'is_trained') and not self.hmm.is_trained:
            print("⚠️ HMM appears not to be trained. Results may be unreliable.")
        
    def _build_road_graph(self) -> nx.Graph:
        """Build NetworkX graph from road segments"""
        G = nx.Graph()
        
        print(f"   🔧 Building road graph from {len(self.segments)} segments...")
        
        for seg_id, segment in self.segments.items():
            # Ensure segment has required attributes
            if not hasattr(segment, 'center') or not hasattr(segment, 'connected_segments'):
                print(f"   ⚠️ Segment {seg_id} missing required attributes")
                continue
                
            G.add_node(seg_id, 
                      pos=segment.center[:2],
                      width=getattr(segment, 'width', 4.0),
                      curvature=getattr(segment, 'curvature', 0.0),
                      density=getattr(segment, 'point_density', 50.0))
            
            for connected_id in segment.connected_segments:
                if connected_id in self.segments:
                    distance = np.linalg.norm(segment.center[:2] - self.segments[connected_id].center[:2])
                    G.add_edge(seg_id, connected_id, weight=distance)
        
        print(f"   ✅ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Check connectivity
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            largest_component_size = max(len(comp) for comp in components) if components else 0
            print(f"   📊 Largest connected component: {largest_component_size} nodes")
            
            if largest_component_size < G.number_of_nodes() * 0.8:
                print(f"   ⚠️ Network is fragmented ({len(components)} components)")
        
        return G
    
    def generate_route_portfolio(self, start_seg: int, end_seg: int, n_routes: int = 8) -> Dict[str, List[int]]:
        """Generate diverse portfolio of routes using different methods"""
        
        routes = {}
        
        print(f"   🔍 Generating routes for network with {len(self.road_graph.nodes)} nodes, {len(self.road_graph.edges)} edges")
        
        # Validate start and end segments
        if start_seg not in self.segments:
            print(f"   ❌ Start segment {start_seg} not in segments")
            return routes
        if end_seg not in self.segments:
            print(f"   ❌ End segment {end_seg} not in segments")
            return routes
        
        # 1. HMM Viterbi route
        print("   🧠 Generating HMM Viterbi route...")
        try:
            if hasattr(self.hmm, 'viterbi_decode'):
                viterbi_route = self.hmm.viterbi_decode(start_seg, end_seg)
                if viterbi_route and len(viterbi_route) > 2:
                    routes['HMM_Viterbi'] = viterbi_route
                    print(f"      ✅ HMM_Viterbi: {len(viterbi_route)} segments")
                else:
                    print("      ❌ HMM_Viterbi: No valid route generated")
            else:
                print("      ❌ HMM does not have viterbi_decode method")
        except Exception as e:
            print(f"      ❌ HMM_Viterbi failed: {e}")
        
        # 2. HMM Forward sampling routes (only if HMM has required attributes)
        if hasattr(self.hmm, 'segment_to_state') and hasattr(self.hmm, 'transition_probs'):
            print("   🎲 Generating HMM sampling routes...")
            for i in range(min(3, n_routes-1)):  # Limit sampling attempts
                try:
                    sampled_route = self._hmm_forward_sample(start_seg, end_seg)
                    if sampled_route and len(sampled_route) > 2:
                        routes[f'HMM_Sample_{i+1}'] = sampled_route
                        print(f"      ✅ HMM_Sample_{i+1}: {len(sampled_route)} segments")
                except Exception as e:
                    print(f"      ❌ HMM_Sample_{i+1} failed: {e}")
        else:
            print("   ⚠️ Skipping HMM sampling - missing required attributes")
        
        # 3. Shortest path (baseline)
        print("   🎯 Generating shortest path...")
        try:
            if self.road_graph.has_node(start_seg) and self.road_graph.has_node(end_seg):
                shortest_route = nx.shortest_path(self.road_graph, start_seg, end_seg, weight='weight')
                if len(shortest_route) > 1:
                    routes['Shortest_Path'] = shortest_route
                    print(f"      ✅ Shortest_Path: {len(shortest_route)} segments")
                else:
                    print("      ❌ Shortest path too short")
            else:
                print("      ❌ Start or end not in road graph")
        except nx.NetworkXNoPath:
            print("      ❌ No path exists between start and end")
        except Exception as e:
            print(f"      ❌ Shortest_Path failed: {e}")
        
        # 4. Alternative paths - IMPROVED WITH TIMEOUT AND LIMITS
        print("   🔀 Generating alternative paths...")
        try:
            # Use more conservative limits for large networks
            max_paths = 2 if len(self.road_graph.nodes) > 100 else 3
            timeout = 5.0 if len(self.road_graph.nodes) > 200 else 10.0
            
            alt_routes = self._safe_alternative_paths(start_seg, end_seg, max_paths=max_paths, timeout=timeout)
            
            for i, route in enumerate(alt_routes):
                route_name = f'Alternative_{i+1}'
                routes[route_name] = route
                print(f"      ✅ {route_name}: {len(route)} segments")
                
        except Exception as e:
            print(f"      ❌ Alternative paths failed: {e}")
        
        # 5. Width-biased routes (prefer wider roads)
        print("   🛣️ Generating width-biased route...")
        try:
            width_route = self._width_biased_route(start_seg, end_seg)
            if width_route and len(width_route) > 1:
                routes['Width_Biased'] = width_route
                print(f"      ✅ Width_Biased: {len(width_route)} segments")
            else:
                print("      ❌ Width-biased route failed")
        except Exception as e:
            print(f"      ❌ Width_Biased failed: {e}")
        
        # 6. Curvature-minimizing routes (prefer straight roads)
        print("   📏 Generating straight-preferred route...")
        try:
            straight_route = self._curvature_minimizing_route(start_seg, end_seg)
            if straight_route and len(straight_route) > 1:
                routes['Straight_Preferred'] = straight_route
                print(f"      ✅ Straight_Preferred: {len(straight_route)} segments")
            else:
                print("      ❌ Straight-preferred route failed")
        except Exception as e:
            print(f"      ❌ Straight_Preferred failed: {e}")
        
        if not routes:
            print("   ⚠️ No routes generated successfully")
        
        return routes
    
    def _safe_alternative_paths(self, start_seg: int, end_seg: int, max_paths: int = 2, timeout: float = 5.0) -> List[List[int]]:
        """Safely find alternative paths with strict timeout and limits"""
        
        if not (self.road_graph.has_node(start_seg) and self.road_graph.has_node(end_seg)):
            return []
        
        alternative_routes = []
        start_time = time.time()
        
        try:
            # Use very conservative cutoff for large networks
            network_size = len(self.road_graph.nodes)
            if network_size > 200:
                cutoff = 4
            elif network_size > 100:
                cutoff = 6
            else:
                cutoff = 8
            
            path_count = 0
            
            # Use generator with early termination
            path_generator = nx.all_simple_paths(self.road_graph, start_seg, end_seg, cutoff=cutoff)
            
            for path in path_generator:
                # Strict timeout check
                if time.time() - start_time > timeout:
                    print(f"      ⏰ Alternative path search timed out after {timeout}s")
                    break
                
                # Check path length constraints
                if 2 < len(path) <= 10:  # Reasonable path length
                    alternative_routes.append(path)
                    path_count += 1
                    
                    # Strict limit on number of paths
                    if path_count >= max_paths:
                        break
            
        except Exception as e:
            print(f"      ⚠️ Alternative path search error: {e}")
        
        return alternative_routes
    
    def _hmm_forward_sample(self, start_seg: int, end_seg: int, max_length: int = 12) -> List[int]:
        """Enhanced forward sampling with better termination"""
        
        if not hasattr(self.hmm, 'segment_to_state') or start_seg not in self.hmm.segment_to_state:
            return []
        
        current_state = self.hmm.segment_to_state[start_seg]
        end_state = self.hmm.segment_to_state.get(end_seg, -1)
        
        if not hasattr(self.hmm, 'transition_probs') or current_state >= len(self.hmm.transition_probs):
            return []
        
        route_states = [current_state]
        
        for step in range(max_length - 1):
            try:
                # Get transition probabilities
                probs = self.hmm.transition_probs[current_state].copy()
                
                # Add bias toward end state as we get longer
                if end_state != -1 and end_state < len(probs) and step > max_length // 3:
                    bias_strength = min(2.0, step / (max_length // 3))
                    probs[end_state] *= bias_strength
                
                # Normalize and sample
                probs = probs / (probs.sum() + 1e-10)
                next_state = np.random.choice(len(probs), p=probs)
                route_states.append(next_state)
                
                if next_state == end_state:
                    break
                
                current_state = next_state
                
            except Exception as e:
                break
        
        # Convert to segments
        route_segments = []
        for state in route_states:
            if hasattr(self.hmm, 'state_to_segment') and state in self.hmm.state_to_segment:
                route_segments.append(self.hmm.state_to_segment[state])
        
        return route_segments
    
    def _width_biased_route(self, start_seg: int, end_seg: int) -> List[int]:
        """Generate route preferring wider roads"""
        
        if not (self.road_graph.has_node(start_seg) and self.road_graph.has_node(end_seg)):
            return []
        
        # Create weight function favoring wider roads
        def width_weight(u, v, d):
            try:
                u_width = getattr(self.segments[u], 'width', 4.0)
                v_width = getattr(self.segments[v], 'width', 4.0)
                avg_width = (u_width + v_width) / 2
                
                # Lower weight for wider roads
                distance = d.get('weight', 1.0)
                width_bonus = max(0, 4.0 - avg_width)  # Penalty for narrow roads
                return distance + width_bonus * 1.5
            except:
                return d.get('weight', 1.0)
        
        try:
            return nx.shortest_path(self.road_graph, start_seg, end_seg, weight=width_weight)
        except:
            return []
    
    def _curvature_minimizing_route(self, start_seg: int, end_seg: int) -> List[int]:
        """Generate route minimizing curvature (straighter paths)"""
        
        if not (self.road_graph.has_node(start_seg) and self.road_graph.has_node(end_seg)):
            return []
        
        def curvature_weight(u, v, d):
            try:
                u_curvature = getattr(self.segments[u], 'curvature', 0.0)
                v_curvature = getattr(self.segments[v], 'curvature', 0.0)
                avg_curvature = (u_curvature + v_curvature) / 2
                
                # Higher weight for more curved roads
                distance = d.get('weight', 1.0)
                curvature_penalty = avg_curvature * 5.0
                return distance + curvature_penalty
            except:
                return d.get('weight', 1.0)
        
        try:
            return nx.shortest_path(self.road_graph, start_seg, end_seg, weight=curvature_weight)
        except:
            return []

class RouteEvaluator:
    """Comprehensive route evaluation and comparison"""
    
    def __init__(self, segments: Dict, road_graph: nx.Graph):
        self.segments = segments
        self.road_graph = road_graph
    
    def evaluate_route(self, route: List[int], route_id: str) -> RouteMetrics:
        """Comprehensive route evaluation with error handling"""
        
        if len(route) < 2:
            return RouteMetrics(route_id, route, 0, 0, 0, 0, 0, 0)
        
        try:
            # Calculate metrics with error handling
            total_length = self._calculate_route_length(route)
            efficiency_ratio = self._calculate_efficiency_ratio(route)
            curvature_score = self._calculate_curvature_score(route)
            width_variance = self._calculate_width_variance(route)
            connectivity_score = self._calculate_connectivity_score(route)
            diversity_score = self._calculate_diversity_score(route)
            
            return RouteMetrics(
                route_id=route_id,
                segments=route,
                total_length=total_length,
                efficiency_ratio=efficiency_ratio,
                curvature_score=curvature_score,
                width_variance=width_variance,
                connectivity_score=connectivity_score,
                diversity_score=diversity_score
            )
        except Exception as e:
            print(f"   ⚠️ Error evaluating route {route_id}: {e}")
            return RouteMetrics(route_id, route, 0, 0, 0, 0, 0, 0)
    
    def _calculate_route_length(self, route: List[int]) -> float:
        """Calculate total route length"""
        total_length = 0
        for i in range(len(route) - 1):
            if route[i] in self.segments and route[i+1] in self.segments:
                try:
                    seg1_center = self.segments[route[i]].center[:2]
                    seg2_center = self.segments[route[i+1]].center[:2]
                    total_length += np.linalg.norm(seg2_center - seg1_center)
                except:
                    pass
        return total_length
    
    def _calculate_efficiency_ratio(self, route: List[int]) -> float:
        """Calculate efficiency vs shortest path"""
        if len(route) < 2:
            return 0
        
        start_seg, end_seg = route[0], route[-1]
        route_length = self._calculate_route_length(route)
        
        if route_length == 0:
            return 0
        
        try:
            if self.road_graph.has_node(start_seg) and self.road_graph.has_node(end_seg):
                shortest_path = nx.shortest_path(self.road_graph, start_seg, end_seg, weight='weight')
                shortest_length = self._calculate_route_length(shortest_path)
                
                if shortest_length > 0:
                    return min(1.0, shortest_length / route_length)  # Cap at 1.0
        except:
            pass
        
        return 1.0  # Default if shortest path can't be calculated
    
    def _calculate_curvature_score(self, route: List[int]) -> float:
        """Calculate average curvature along route"""
        curvatures = []
        for seg_id in route:
            if seg_id in self.segments:
                try:
                    curvature = getattr(self.segments[seg_id], 'curvature', 0.0)
                    curvatures.append(curvature)
                except:
                    pass
        
        return np.mean(curvatures) if curvatures else 0
    
    def _calculate_width_variance(self, route: List[int]) -> float:
        """Calculate variance in road width along route"""
        widths = []
        for seg_id in route:
            if seg_id in self.segments:
                try:
                    width = getattr(self.segments[seg_id], 'width', 4.0)
                    widths.append(width)
                except:
                    pass
        
        return np.var(widths) if len(widths) > 1 else 0
    
    def _calculate_connectivity_score(self, route: List[int]) -> float:
        """Calculate average connectivity along route"""
        connectivities = []
        for seg_id in route:
            if seg_id in self.segments:
                try:
                    connectivity = len(getattr(self.segments[seg_id], 'connected_segments', []))
                    connectivities.append(connectivity)
                except:
                    pass
        
        return np.mean(connectivities) if connectivities else 0
    
    def _calculate_diversity_score(self, route: List[int]) -> float:
        """Calculate route diversity (how much it varies from typical patterns)"""
        if len(route) < 3:
            return 0
        
        # Calculate directional changes
        direction_changes = 0
        valid_comparisons = 0
        
        for i in range(len(route) - 2):
            if all(seg_id in self.segments for seg_id in route[i:i+3]):
                try:
                    p1 = self.segments[route[i]].center[:2]
                    p2 = self.segments[route[i+1]].center[:2]
                    p3 = self.segments[route[i+2]].center[:2]
                    
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        
                        if angle > np.pi / 4:  # 45 degree threshold
                            direction_changes += 1
                        valid_comparisons += 1
                except:
                    pass
        
        return direction_changes / max(1, valid_comparisons)

class RouteComparator:
    """Compare and analyze multiple routes"""
    
    def __init__(self, evaluator: RouteEvaluator):
        self.evaluator = evaluator
    
    def compare_routes(self, routes: Dict[str, List[int]]) -> pd.DataFrame:
        """Compare multiple routes and return analysis DataFrame"""
        
        if not routes:
            return pd.DataFrame()
        
        metrics_list = []
        
        for route_id, route in routes.items():
            try:
                metrics = self.evaluator.evaluate_route(route, route_id)
                metrics_list.append({
                    'Route_ID': metrics.route_id,
                    'Segments': len(metrics.segments),
                    'Length': metrics.total_length,
                    'Efficiency': metrics.efficiency_ratio,
                    'Avg_Curvature': metrics.curvature_score,
                    'Width_Variance': metrics.width_variance,
                    'Avg_Connectivity': metrics.connectivity_score,
                    'Diversity_Score': metrics.diversity_score
                })
            except Exception as e:
                print(f"   ⚠️ Error comparing route {route_id}: {e}")
                continue
        
        return pd.DataFrame(metrics_list)
    
    def route_similarity_matrix(self, routes: Dict[str, List[int]]) -> Tuple[np.ndarray, List[str]]:
        """Calculate pairwise route similarity matrix"""
        
        route_ids = list(routes.keys())
        n_routes = len(route_ids)
        
        if n_routes == 0:
            return np.array([]), []
        
        similarity_matrix = np.zeros((n_routes, n_routes))
        
        try:
            for i in range(n_routes):
                for j in range(n_routes):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        route1 = set(routes[route_ids[i]])
                        route2 = set(routes[route_ids[j]])
                        
                        # Jaccard similarity
                        intersection = len(route1 & route2)
                        union = len(route1 | route2)
                        similarity_matrix[i, j] = intersection / union if union > 0 else 0
        except Exception as e:
            print(f"   ⚠️ Error calculating similarity matrix: {e}")
        
        return similarity_matrix, route_ids

def create_comprehensive_analysis(hmm, segments, start_seg: int, end_seg: int):
    """Create comprehensive route analysis with enhanced error handling"""
    
    print(f"\n🔬 Comprehensive Route Analysis: {start_seg} → {end_seg}")
    print("=" * 60)
    
    # Input validation
    if not segments:
        print("   ❌ No segments provided!")
        return
    
    if start_seg not in segments or end_seg not in segments:
        print(f"   ❌ Invalid segment IDs: {start_seg} or {end_seg} not in segments")
        return
    
    try:
        # 1. Generate diverse routes
        print("1. 🛣️ Generating diverse route portfolio...")
        generator = AdvancedRouteGenerator(hmm, segments)
        routes = generator.generate_route_portfolio(start_seg, end_seg, n_routes=8)
        
        print(f"\n   ✅ Generated {len(routes)} different routes:")
        for route_id, route in routes.items():
            print(f"      - {route_id}: {len(route)} segments")
        
        if not routes:
            print("   ❌ No routes generated! Analysis cannot continue.")
            return
        
        # 2. Evaluate routes
        print("\n2. 📊 Evaluating route quality...")
        road_graph = generator.road_graph
        evaluator = RouteEvaluator(segments, road_graph)
        comparator = RouteComparator(evaluator)
        
        comparison_df = comparator.compare_routes(routes)
        
        if not comparison_df.empty:
            print("   ✅ Route comparison:")
            print(comparison_df.round(3))
        else:
            print("   ⚠️ No routes could be evaluated")
        
        # 3. Create visualizations
        print("\n3. 📈 Creating comprehensive visualizations...")
        create_route_analysis_plots(routes, segments, comparison_df, comparator)
        
        # 4. Generate insights
        print("\n4. 💡 Route Analysis Insights:")
        generate_route_insights(comparison_df, routes)
        
    except Exception as e:
        print(f"   ❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

def create_route_analysis_plots(routes: Dict, segments: Dict, comparison_df: pd.DataFrame, comparator: RouteComparator):
    """Create comprehensive route analysis visualizations with error handling"""
    
    if len(routes) == 0:
        print("   ⚠️ No routes to visualize")
        return
    
    try:
        # Create figure with error handling
        fig = plt.figure(figsize=(20, 15))
        
        # Calculate number of plots needed
        n_plots = 3 if not comparison_df.empty else 6
        
        if n_plots == 3:
            # Simple layout for limited data
            ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((2, 2), (1, 0))
            ax3 = plt.subplot2grid((2, 2), (1, 1))
            axes = [ax1, ax2, ax3]
        else:
            # Full layout
            ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
            ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
            ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
            ax4 = plt.subplot2grid((3, 4), (2, 0))
            ax5 = plt.subplot2grid((3, 4), (2, 1))
            ax6 = plt.subplot2grid((3, 4), (2, 2))
            axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        
        # Plot 1: All routes on map
        ax = axes[0]
        
        # Draw all segments lightly
        segment_centers = []
        for seg_id, segment in segments.items():
            try:
                center = segment.center
                ax.scatter(center[0], center[1], c='lightgray', s=10, alpha=0.3)
                segment_centers.append([center[0], center[1]])
            except:
                continue
        
        # Draw routes with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(routes), 10)))
        
        for i, (route_id, route) in enumerate(routes.items()):
            if len(route) > 1:
                route_centers = []
                for seg_id in route:
                    if seg_id in segments:
                        try:
                            route_centers.append(segments[seg_id].center)
                        except:
                            continue
                
                if route_centers:
                    route_centers = np.array(route_centers)
                    color = colors[i % len(colors)]
                    
                    ax.plot(route_centers[:, 0], route_centers[:, 1], 
                           color=color, linewidth=3, alpha=0.8, label=route_id)
                    
                    # Mark start and end
                    ax.scatter(route_centers[0, 0], route_centers[0, 1], 
                              color=color, s=100, marker='o', edgecolor='black', linewidth=2)
                    ax.scatter(route_centers[-1, 0], route_centers[-1, 1], 
                              color=color, s=100, marker='s', edgecolor='black', linewidth=2)
        
        ax.set_title('Route Portfolio Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Additional plots only if we have evaluation data
        if not comparison_df.empty and len(axes) > 3:
            # Plot 2: Route metrics comparison
            ax = axes[1]
            
            metrics_to_plot = ['Length', 'Efficiency', 'Avg_Curvature', 'Avg_Connectivity']
            x_pos = np.arange(len(comparison_df))
            width = 0.2
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in comparison_df.columns:
                    values = comparison_df[metric].values
                    # Normalize values for comparison
                    if len(values) > 0 and values.max() > values.min():
                        normalized_values = (values - values.min()) / (values.max() - values.min())
                    else:
                        normalized_values = values
                    ax.bar(x_pos + i * width, normalized_values, width, label=metric, alpha=0.8)
            
            ax.set_title('Normalized Route Metrics', fontweight='bold')
            ax.set_xlabel('Routes')
            ax.set_ylabel('Normalized Value')
            ax.set_xticks(x_pos + width * 1.5)
            ax.set_xticklabels(comparison_df['Route_ID'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Route similarity heatmap
            ax = axes[2]
            
            try:
                similarity_matrix, route_ids = comparator.route_similarity_matrix(routes)
                
                if similarity_matrix.size > 0:
                    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
                    ax.set_xticks(range(len(route_ids)))
                    ax.set_yticks(range(len(route_ids)))
                    ax.set_xticklabels(route_ids, rotation=45, ha='right')
                    ax.set_yticklabels(route_ids)
                    ax.set_title('Route Similarity Matrix', fontweight='bold')
                    
                    # Add text annotations for small matrices
                    if len(route_ids) <= 8:
                        for i in range(len(route_ids)):
                            for j in range(len(route_ids)):
                                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
                    
                    plt.colorbar(im, ax=ax, label='Similarity Score')
                else:
                    ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Route Similarity Matrix', fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Route Similarity Matrix (Error)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")
        print("   📊 Creating simple route summary instead...")
        
        # Fallback: simple text summary
        print(f"\n   📋 Route Summary:")
        for route_id, route in routes.items():
            print(f"      {route_id}: {len(route)} segments")

def generate_route_insights(comparison_df: pd.DataFrame, routes: Dict):
    """Generate textual insights about the routes with error handling"""
    
    if comparison_df.empty:
        print("   ⚠️ No route evaluation data available for insights")
        print(f"   📊 Generated {len(routes)} routes with the following lengths:")
        for route_id, route in routes.items():
            print(f"      - {route_id}: {len(route)} segments")
        return
    
    try:
        # Find best routes for different criteria
        if 'Efficiency' in comparison_df.columns and not comparison_df['Efficiency'].isna().all():
            best_efficiency = comparison_df.loc[comparison_df['Efficiency'].idxmax()]
            print(f"   🏆 Most Efficient Route: {best_efficiency['Route_ID']} (efficiency: {best_efficiency['Efficiency']:.3f})")
        
        if 'Diversity_Score' in comparison_df.columns and not comparison_df['Diversity_Score'].isna().all():
            best_diversity = comparison_df.loc[comparison_df['Diversity_Score'].idxmax()]
            print(f"   🌟 Most Diverse Route: {best_diversity['Route_ID']} (diversity: {best_diversity['Diversity_Score']:.3f})")
        
        if 'Length' in comparison_df.columns and not comparison_df['Length'].isna().all():
            shortest_route = comparison_df.loc[comparison_df['Length'].idxmin()]
            print(f"   🎯 Shortest Route: {shortest_route['Route_ID']} (length: {shortest_route['Length']:.1f}m)")
        
        if 'Avg_Connectivity' in comparison_df.columns and not comparison_df['Avg_Connectivity'].isna().all():
            most_connected = comparison_df.loc[comparison_df['Avg_Connectivity'].idxmax()]
            print(f"   🔗 Most Connected Route: {most_connected['Route_ID']} (connectivity: {most_connected['Avg_Connectivity']:.1f})")
        
        # HMM vs baseline comparison
        hmm_routes = comparison_df[comparison_df['Route_ID'].str.contains('HMM', na=False)]
        baseline_routes = comparison_df[~comparison_df['Route_ID'].str.contains('HMM', na=False)]
        
        if len(hmm_routes) > 0 and len(baseline_routes) > 0:
            print(f"\n   📊 HMM vs Baseline Comparison:")
            
            if 'Efficiency' in comparison_df.columns:
                hmm_eff = hmm_routes['Efficiency'].mean()
                baseline_eff = baseline_routes['Efficiency'].mean()
                print(f"      HMM Average Efficiency: {hmm_eff:.3f}")
                print(f"      Baseline Average Efficiency: {baseline_eff:.3f}")
            
            if 'Diversity_Score' in comparison_df.columns:
                hmm_div = hmm_routes['Diversity_Score'].mean()
                baseline_div = baseline_routes['Diversity_Score'].mean()
                print(f"      HMM Average Diversity: {hmm_div:.3f}")
                print(f"      Baseline Average Diversity: {baseline_div:.3f}")
        
        # Route similarity insights
        total_routes = len(routes)
        unique_segments = set()
        for route in routes.values():
            unique_segments.update(route)
        
        print(f"\n   🗺️ Portfolio Coverage:")
        print(f"      Total unique segments used: {len(unique_segments)}")
        
        if 'Segments' in comparison_df.columns:
            print(f"      Average route length: {comparison_df['Segments'].mean():.1f} segments")
            print(f"      Route length range: {comparison_df['Segments'].min()}-{comparison_df['Segments'].max()} segments")
        
    except Exception as e:
        print(f"   ⚠️ Error generating insights: {e}")
        print(f"   📊 Basic summary: {len(routes)} routes generated")

# REMOVED MOCK CLASSES TO AVOID CONFLICTS
# The mock classes have been removed to prevent conflicts with real classes
# from the main KITTI analysis. This module now works purely as an analysis
# extension for real HMM and segment data.

def validate_inputs(hmm, segments, start_seg: int, end_seg: int) -> bool:
    """Validate inputs before running analysis"""
    
    if not segments:
        print("❌ No segments provided")
        return False
    
    if start_seg not in segments:
        print(f"❌ Start segment {start_seg} not found in segments")
        return False
    
    if end_seg not in segments:
        print(f"❌ End segment {end_seg} not found in segments")
        return False
    
    if not hasattr(hmm, 'n_states'):
        print("❌ HMM missing n_states attribute")
        return False
    
    return True

def main():
    """Main function for standalone testing (with real data only)"""
    
    print("🔬 ADVANCED HMM ROUTE ANALYSIS")
    print("=" * 50)
    print("⚠️ This module requires real HMM and segment data")
    print("   Import this module from your main KITTI analysis")
    print("   or call create_comprehensive_analysis() directly")
    print("\nUsage:")
    print("   import hmm_advanced_analysis")
    print("   hmm_advanced_analysis.create_comprehensive_analysis(hmm, segments, start_seg, end_seg)")

if __name__ == "__main__":
    main()