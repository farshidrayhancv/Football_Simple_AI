"""Simple player possession detector for football analysis."""

import numpy as np
import cv2
import supervision as sv


class PlayerPossessionDetector:
    def __init__(self, proximity_threshold=50, possession_frames=3):
        """Initialize player possession detector.
        
        Args:
            proximity_threshold: Distance in pixels for a player to be considered in possession
            possession_frames: Number of frames a player needs to be closest to be in possession
        """
        self.proximity_threshold = proximity_threshold
        self.possession_frames = possession_frames
        
        # Possession tracking
        self.current_possession = None  # player_id
        self.current_team = None  # team_id
        self.closest_candidate = None  # Temporary tracking of closest player
        self.candidate_counter = 0  # Counter to track consecutive frames with same player closest
        
        print(f"PlayerPossessionDetector initialized with threshold={proximity_threshold}, frames={possession_frames}")
    
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
                'team_id': self.current_team
            }
        
        # Find the closest player to the ball
        closest_player = self._find_closest_player(detections, ball_position)
        
        # If no player is close enough, no one has possession
        if closest_player is None:
            self.candidate_counter = 0
            self.closest_candidate = None
            return {
                'player_id': self.current_possession,
                'team_id': self.current_team
            }
        
        # Get player details
        team_id, player_id, distance = closest_player
        
        # Check if we have a new closest player
        if self.closest_candidate is None or self.closest_candidate != player_id:
            # Reset counter for new player
            self.closest_candidate = player_id
            self.candidate_counter = 1
        else:
            # Same player is still closest, increment counter
            self.candidate_counter += 1
            
            # If player has been closest for enough frames, they have possession
            if self.candidate_counter >= self.possession_frames:
                self.current_possession = player_id
                self.current_team = team_id
                print(f"Player #{player_id} (team {team_id}) now has possession")
        
        return {
            'player_id': self.current_possession,
            'team_id': self.current_team
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