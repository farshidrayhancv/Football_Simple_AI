"""Pitch rendering module."""

import cv2
import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
    draw_paths_on_pitch
)


class PitchRenderer:
    def __init__(self, config):
        self.config = config
        self.pitch_config = SoccerPitchConfiguration()
    
    def render(self, detections, transformer, ball_trail=None):
        """Render the pitch view."""
        pitch_view = draw_pitch(self.pitch_config)
        
        if transformer is None:
            return pitch_view
        
        # Draw players and goalkeepers
        self._draw_players(pitch_view, detections, transformer)
        
        # Draw referees
        self._draw_referees(pitch_view, detections, transformer)
        
        # Draw ball and trail
        self._draw_ball(pitch_view, detections, transformer, ball_trail)
        
        return pitch_view
    
    def _draw_players(self, pitch_view, detections, transformer):
        """Draw players on pitch."""
        players = detections['players']
        goalkeepers = detections['goalkeepers']
        
        # Combine players and goalkeepers
        all_players = []
        if len(players) > 0:
            all_players.append(players)
        if len(goalkeepers) > 0:
            all_players.append(goalkeepers)
        
        if not all_players:
            return
        
        merged = sv.Detections.merge(all_players)
        if len(merged) == 0:
            return
        
        # Transform coordinates
        xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(points=xy)
        
        # Draw by team
        for team_id in [0, 1]:
            team_mask = merged.class_id == team_id
            if np.any(team_mask):
                color = sv.Color.from_hex(
                    self.config['display']['team_colors'][f'team_{team_id + 1}']
                )
                pitch_view = draw_points_on_pitch(
                    config=self.pitch_config,
                    xy=pitch_xy[team_mask],
                    face_color=color,
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch_view
                )
    
    def _draw_referees(self, pitch_view, detections, transformer):
        """Draw referees on pitch."""
        referees = detections['referees']
        if len(referees) == 0:
            return
        
        xy = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(points=xy)
        
        color = sv.Color.from_hex(self.config['display']['referee_color'])
        pitch_view = draw_points_on_pitch(
            config=self.pitch_config,
            xy=pitch_xy,
            face_color=color,
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch_view
        )
    
    def _draw_ball(self, pitch_view, detections, transformer, ball_trail):
        """Draw ball and trail on pitch."""
        # Draw ball trail
        if ball_trail and len(ball_trail) > 1 and self.config['ball_tracking']['enable']:
            try:
                pitch_view = draw_paths_on_pitch(
                    config=self.pitch_config,
                    paths=[np.array(ball_trail)],
                    color=sv.Color.WHITE,
                    pitch=pitch_view
                )
            except:
                pass  # Skip trail if error
        
        # Draw current ball position
        ball = detections['ball']
        if len(ball) > 0:
            xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_xy = transformer.transform_points(points=xy)
            
            pitch_view = draw_points_on_pitch(
                config=self.pitch_config,
                xy=pitch_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=pitch_view
            )
        
        return pitch_view
