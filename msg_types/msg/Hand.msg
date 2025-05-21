# Represents a single detected hand
string label # Handedness: "Left" or "Right"
int32 hand_id # Index of the hand if multiple are detected (e.g., 0, 1)
Landmark[] landmarks # List of 3D landmarks for this hand