"""Team resolution module."""

import numpy as np
import supervision as sv


class TeamResolver:
    def resolve_goalkeeper_teams(self, players, goalkeepers):
        """Assign goalkeepers to teams based on proximity."""
        if len(goalkeepers) == 0:
            return goalkeepers
        
        if len(players) == 0:
            goalkeepers.class_id = np.zeros(len(goalkeepers), dtype=int)
            return goalkeepers
        
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Get team centroids
        team_0_players = players_xy[players.class_id == 0]
        team_1_players = players_xy[players.class_id == 1]
        
        if len(team_0_players) == 0 or len(team_1_players) == 0:
            goalkeepers.class_id = np.zeros(len(goalkeepers), dtype=int)
            return goalkeepers
        
        team_0_centroid = team_0_players.mean(axis=0)
        team_1_centroid = team_1_players.mean(axis=0)
        
        # Assign based on distance
        goalkeeper_teams = []
        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
            dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
            goalkeeper_teams.append(0 if dist_0 < dist_1 else 1)
        
        goalkeepers.class_id = np.array(goalkeeper_teams, dtype=int)
        return goalkeepers
