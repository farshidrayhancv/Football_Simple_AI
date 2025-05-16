"""Simple player possession detector for football analysis."""

import numpy as np
import cv2
import supervision as sv


class PlayerPossessionDetector:
    def __init__(self, proximity_threshold=50, possession_frames=3, possession_duration=10, no_possession_frames=10):
        """Initialize player possession detector.
        
        Args:
            proximity_threshold: Distance in pixels for a player to be considered in proximity
            possession_frames: Number of frames a player needs to be closest to be considered for possession
            possession_duration: Minimum duration (in frames) a player must maintain possession
                                 to filter out false positives during passes
            no_possession_frames: Number of frames the ball has to be alone to consider no possession
        """
        # For pitch coordinates, we need a much larger threshold
        # Multiplying by 20 because pitch coordinates are in a much larger scale
        self.proximity_threshold = proximity_threshold 
        self.possession_frames = possession_frames
        self.possession_duration = possession_duration
        self.no_possession_frames = no_possession_frames
        
        # Possession tracking
        self.current_possession = None  # player_id
        self.current_team = None  # team_id
        self.closest_candidate = None  # Temporary tracking of closest player
        self.candidate_counter = 0  # Counter to track consecutive frames with same player closest
        self.possession_timer = 0  # Duration counter for confirmed possession
        self.ball_alone_counter = 0  # Counter for how many frames the ball has been alone
        
        # Keep track of recent proximity changes
        self.previous_closest_players = []  # List of recently close players
        self.player_proximity_durations = {}  # Track how long each player has been close to ball
        
        print(f"PlayerPossessionDetector initialized with threshold={self.proximity_threshold} (for pitch coordinates), "
              f"frames={possession_frames}, duration={possession_duration}, no_possession={no_possession_frames}")
    
    def update(self, detections, ball_position):
        """Update possession detection based on current detections.
        
        Args:
            detections: Dictionary of detections for players, goalkeepers, etc.
            ball_position: Position of the ball (x, y) in coordinates
            
        Returns:
            Dictionary with possession detection results
        """
        # If no ball detected, keep current possession but return without updating
        if ball_position is None or len(ball_position) == 0:
            return {
                'player_id': self.current_possession,
                'team_id': self.current_team,
                'status': 'no_ball_detected'
            }
        
        # Find the closest player to the ball
        closest_player = self._find_closest_player(detections, ball_position)
        
        # If no player is close enough, increment ball_alone_counter
        if closest_player is None:
            self.candidate_counter = 0
            self.closest_candidate = None
            self.ball_alone_counter += 1
            
            # Clear possession if ball has been alone for too long
            if self.ball_alone_counter >= self.no_possession_frames:
                if self.current_possession is not None:
                    print(f"Possession cleared: ball alone for {self.ball_alone_counter} frames")
                    self.current_possession = None
                    self.current_team = None
                
            # Clear all player proximity durations
            self.player_proximity_durations = {}
                
            return {
                'player_id': self.current_possession,
                'team_id': self.current_team,
                'status': f'no_player_close (alone for {self.ball_alone_counter} frames)',
                'ball_alone_counter': self.ball_alone_counter
            }
        
        # Reset ball_alone_counter since a player is close to the ball
        self.ball_alone_counter = 0
        
        # Get player details
        team_id, player_id, distance = closest_player
        
        # Track this player's proximity duration
        if player_id not in self.player_proximity_durations:
            self.player_proximity_durations[player_id] = 0
        self.player_proximity_durations[player_id] += 1
        
        # Decay proximity duration for other players
        for pid in list(self.player_proximity_durations.keys()):
            if pid != player_id:
                self.player_proximity_durations[pid] = max(0, self.player_proximity_durations[pid] - 1)
                if self.player_proximity_durations[pid] == 0:
                    self.player_proximity_durations.pop(pid)
        
        # Check if we have a new closest player
        if self.closest_candidate is None or self.closest_candidate != player_id:
            # Reset counter for new player
            self.closest_candidate = player_id
            self.candidate_counter = 1
            
            # Keep previous possession until new possession is confirmed
            status = 'new_closest_player'
        else:
            # Same player is still closest, increment counter
            self.candidate_counter += 1
            status = 'proximity_building'
            
            # If player has been closest for enough frames, they are a possession candidate
            if self.candidate_counter >= self.possession_frames:
                # Check if this is a new player or the same as current possession
                if self.current_possession != player_id:
                    # Only assign new possession if player has been in proximity for long enough
                    if self.player_proximity_durations[player_id] >= self.possession_duration:
                        self.current_possession = player_id
                        self.current_team = team_id
                        self.possession_timer = self.possession_duration  # Reset possession timer
                        status = 'possession_changed'
                        print(f"Player #{player_id} (team {team_id}) now has possession "
                              f"after {self.player_proximity_durations[player_id]} frames of proximity")
                    else:
                        status = 'potential_possession'
                else:
                    # Same player maintains possession, refresh timer
                    self.possession_timer = self.possession_duration
                    status = 'possession_maintained'
        
        # Add status info to return value for logging
        return {
            'player_id': self.current_possession,
            'team_id': self.current_team,
            'status': status,
            'proximity_player': player_id,
            'proximity_duration': self.player_proximity_durations.get(player_id, 0),
            'candidate_counter': self.candidate_counter,
            'possession_timer': self.possession_timer,
            'ball_alone_counter': self.ball_alone_counter
        }
    
    def _find_closest_player(self, detections, ball_position):
        """Find the player closest to the ball."""
        closest_distance = float('inf')
        closest_player = None
        
        # Process players, goalkeepers, and referees
        for player_type in ['players', 'goalkeepers', 'referees']:
            if player_type not in detections or len(detections[player_type]) == 0:
                continue
                
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
        
        if closest_player:
            team_id, player_id, distance = closest_player
            print(f"Closest to ball: Player #{player_id} (team {team_id}) at distance {distance:.2f}")
            
        return closest_player
    
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
            player_id = self.current_possession
            team_id = self.current_team
            
            highlighted = False
            
            # Find this player in the detections
            for player_type in ['players', 'goalkeepers', 'referees']:
                if player_type not in detections or len(detections[player_type]) == 0:
                    continue
                    
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
                            
                            # Draw a distinctive highlight - THICKER BORDER
                            cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 5)
                            
                            # Draw a circle above the player
                            center_x = (box[0] + box[2]) // 2
                            center_y = box[1] - 30
                            cv2.circle(vis_frame, (center_x, center_y), 20, (0, 255, 255), -1)
                            
                            # Add player ID in the circle
                            cv2.putText(vis_frame, f"{player_id}", (center_x - 10, center_y + 7),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
                            # Draw "POSSESSION" label
                            cv2.putText(vis_frame, "POSSESSION", (box[0], box[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                       
                            highlighted = True
                        except Exception as e:
                            print(f"Error highlighting player: {e}")
            
            if highlighted:
                print(f"Successfully highlighted player #{player_id} with possession")
            else:
                print(f"Could not find player #{player_id} to highlight")
        
        return vis_frame
    
    def draw_debug_info(self, frame, detections):
        """Draw debug information about possession status.
        
        Args:
            frame: Frame to draw on
            detections: Current detections
            
        Returns:
            Frame with debug information
        """
        vis_frame = frame.copy()
        
        # Create semi-transparent overlay for stats panel
        height, width = vis_frame.shape[:2]
        panel_height = 220
        panel_width = 300
        panel_x = width - panel_width - 10
        panel_y = 10
        
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0)
        
        # Draw title
        cv2.putText(vis_frame, "POSSESSION DEBUG", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw possession information
        y_offset = panel_y + 50
        
        if self.current_possession is not None:
            cv2.putText(vis_frame, f"Current: Player #{self.current_possession}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(vis_frame, "Current: None", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(vis_frame, f"Candidate: {self.closest_candidate}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(vis_frame, f"Counter: {self.candidate_counter}/{self.possession_frames}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(vis_frame, f"Pos timer: {self.possession_timer}/{self.possession_duration}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        # Add ball alone counter
        cv2.putText(vis_frame, f"Ball alone: {self.ball_alone_counter}/{self.no_possession_frames}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        # Show threshold info
        cv2.putText(vis_frame, f"Proximity threshold: {self.proximity_threshold:.1f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        # Show proximity durations for players
        if self.player_proximity_durations:
            cv2.putText(vis_frame, "Proximity durations:", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 25
            
            # Sort by duration
            sorted_durations = sorted(
                self.player_proximity_durations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for player_id, duration in sorted_durations[:3]:  # Show top 3
                cv2.putText(vis_frame, f"  #{player_id}: {duration} frames", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_offset += 20
        
        return vis_frame