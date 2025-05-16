"""Possession detection module for football analysis."""

import numpy as np
import cv2
import supervision as sv


class PossessionDetector:
    def __init__(self, proximity_threshold=50, possession_frames=5):
        """Initialize possession detector.
        
        Args:
            proximity_threshold: Distance in pixels for a player to be considered in possession
            possession_frames: Number of frames a player needs to be closest to be in possession
        """
        self.proximity_threshold = proximity_threshold
        self.possession_frames = possession_frames
        
        # Possession tracking
        self.current_possession = None  # (team_id, player_id)
        self.possession_counter = 0  # Counter to track consecutive frames with same player closest
        self.ball_in_play = False  # Is the ball in play (not off-field)
        
        # Statistics
        self.possession_stats = {}  # Player ID -> {team, frames}
        self.team_possession = {0: 0, 1: 0, 2: 0, 3: 0}  # Team ID -> frames (include refs and others)
        
        # Debug info
        self.debug_info = {}
    
    def update(self, detections, ball_position, frame=None):
        """Update possession detection based on current detections.
        
        Args:
            detections: Dictionary of detections for players, goalkeepers, etc.
            ball_position: Position of the ball (x, y) in pitch coordinates
            frame: Optional frame for visualization
        
        Returns:
            Dictionary with possession detection results
        """
        # If no ball detected, skip processing
        if ball_position is None or len(ball_position) == 0:
            self.debug_info = {'status': 'No ball detected'}
            return self._get_results()
        
        # Find the closest player to the ball
        closest_player = self._find_closest_player(detections, ball_position)
        
        # If no player is close enough, no one has possession
        if closest_player is None:
            self.possession_counter = 0
            self.current_possession = None
            self.debug_info = {'status': 'No player close to ball'}
            return self._get_results()
        
        # Get player details
        team_id, player_id, distance = closest_player
        
        # Initialize player in stats if not exists
        if player_id not in self.possession_stats:
            self.possession_stats[player_id] = {'team': team_id, 'frames': 0}
        
        # Track possession changes
        if self.current_possession is None:
            # First possession - initialize
            if self.possession_counter >= self.possession_frames:
                self.current_possession = (team_id, player_id)
                self._update_stats(team_id, player_id)
                self.debug_info = {'status': f'Initial possession: Team {team_id}, Player {player_id}'}
            else:
                self.possession_counter += 1
                self.debug_info = {'status': f'Building initial possession: {self.possession_counter}/{self.possession_frames}'}
        
        elif self.current_possession[1] == player_id:
            # Same player still has possession
            self.possession_counter = min(self.possession_counter + 1, self.possession_frames * 2)
            self._update_stats(team_id, player_id)
            self.debug_info = {'status': f'Continued possession: Player {player_id}'}
        
        else:
            # Different player is closest - potential possession change
            self.debug_info = {'status': f'Potential possession change: {self.possession_counter}/{self.possession_frames}'}
            
            # Check if the player has been closest for enough frames
            if self.possession_counter >= self.possession_frames:
                # Enough frames have passed to consider a real possession change
                self.current_possession = (team_id, player_id)
                self._update_stats(team_id, player_id)
                self.possession_counter = 0
                self.debug_info = {'status': f'Possession changed: Team {team_id}, Player {player_id}'}
            else:
                # Not enough frames yet, increment counter
                self.possession_counter += 1
        
        return self._get_results()
    
    def _find_closest_player(self, detections, ball_position):
        """Find the player closest to the ball."""
        closest_distance = float('inf')
        closest_player = None
        
        # Process both players and goalkeepers
        for player_type in ['players', 'goalkeepers']:
            if player_type in detections and len(detections[player_type]) > 0:
                if not hasattr(detections[player_type], 'get_anchors_coordinates'):
                    continue
                    
                player_positions = detections[player_type].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                
                if (not hasattr(detections[player_type], 'class_id') or 
                    not hasattr(detections[player_type], 'tracker_id')):
                    continue
                    
                team_ids = detections[player_type].class_id
                tracker_ids = detections[player_type].tracker_id
                
                if tracker_ids is None:
                    continue
                
                for i, (pos, team_id, tracker_id) in enumerate(zip(player_positions, team_ids, tracker_ids)):
                    distance = np.linalg.norm(pos - ball_position)
                    if distance < closest_distance and distance < self.proximity_threshold:
                        closest_distance = distance
                        closest_player = (int(team_id), int(tracker_id), distance)
        
        return closest_player
    
    def _update_stats(self, team_id, player_id):
        """Update possession statistics."""
        # Update player stats
        self.possession_stats[player_id]['frames'] += 1
        
        # Update team stats - ensure team_id exists in dictionary
        if team_id not in self.team_possession:
            self.team_possession[team_id] = 0
        
        self.team_possession[team_id] += 1
    
    def _get_results(self):
        """Get the current possession detection results."""
        return {
            'current_possession': self.current_possession,
            'possession_stats': self.possession_stats,
            'team_possession': self.team_possession,
            'debug_info': self.debug_info
        }
    
    def draw_possession_stats(self, frame):
        """Draw possession statistics on frame.
        
        Args:
            frame: Frame to draw on
        
        Returns:
            Frame with statistics drawn
        """
        # Make a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw background for the stats panel
        height, width = vis_frame.shape[:2]
        panel_height = min(300, height // 2)
        panel_width = 220
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Draw semi-transparent background
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0)
        
        # Draw title
        cv2.putText(vis_frame, "POSSESSION STATISTICS", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate team possession percentages (only for teams 0 and 1)
        team_frames = {0: self.team_possession.get(0, 0), 1: self.team_possession.get(1, 0)}
        total_frames = sum(team_frames.values())
        
        if total_frames > 0:
            team_percentages = {
                team_id: (frames / total_frames) * 100 
                for team_id, frames in team_frames.items()
            }
        else:
            team_percentages = {0: 0, 1: 0}
        
        # Draw team possession stats
        y_offset = panel_y + 70
        
        # Team 1
        team_color = (0, 0, 255)  # Red for team 1
        cv2.putText(vis_frame, f"Team 1: {team_percentages[0]:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
        y_offset += 25
        
        # Team 2
        team_color = (255, 0, 0)  # Blue for team 2
        cv2.putText(vis_frame, f"Team 2: {team_percentages[1]:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
        y_offset += 40
        
        # Draw line separator
        cv2.line(vis_frame, (panel_x + 10, y_offset - 20), (panel_x + panel_width - 10, y_offset - 20),
                (200, 200, 200), 1)
        
        # Current possession
        if self.current_possession is not None:
            team_id, player_id = self.current_possession
            
            # Only display main teams (0 and 1)
            if team_id in [0, 1]:
                team_color = (0, 0, 255) if team_id == 0 else (255, 0, 0)
                
                cv2.putText(vis_frame, "Current Possession:", (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                cv2.putText(vis_frame, f"Player #{player_id} (Team {team_id+1})", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
            else:
                cv2.putText(vis_frame, "Current Possession:", (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                cv2.putText(vis_frame, f"Referee/Other #{player_id}", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "No Current Possession", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def highlight_possession(self, frame, detections):
        """Highlight the player in possession of the ball.
        
        Args:
            frame: Frame to draw on
            detections: Current detections to find player
            
        Returns:
            Frame with possession player highlighted
        """
        vis_frame = frame.copy()
        
        # If there's a current possession, highlight that player
        if self.current_possession is not None:
            team_id, player_id = self.current_possession
            
            # Find this player in the detections
            for player_type in ['players', 'goalkeepers', 'referees']:
                if player_type in detections and len(detections[player_type]) > 0:
                    if not hasattr(detections[player_type], 'tracker_id'):
                        continue
                        
                    tracker_ids = detections[player_type].tracker_id
                    
                    if tracker_ids is None:
                        continue
                    
                    for i, tid in enumerate(tracker_ids):
                        if int(tid) == player_id:
                            # Highlight this player
                            try:
                                box = detections[player_type].xyxy[i].astype(int)
                                
                                # Draw a distinctive highlight
                                cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)
                                
                                # Draw a circle above the player
                                center_x = (box[0] + box[2]) // 2
                                center_y = box[1] - 30
                                cv2.circle(vis_frame, (center_x, center_y), 15, (0, 255, 255), -1)
                                
                                # Add player ID in the circle
                                cv2.putText(vis_frame, f"{player_id}", (center_x - 7, center_y + 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                
                                # Draw "POSSESSION" label
                                cv2.putText(vis_frame, "POSSESSION", (box[0], box[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            except:
                                pass
        
        return vis_frame