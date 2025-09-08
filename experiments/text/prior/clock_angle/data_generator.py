import numpy as np
import random


def random_time_with_angle(target_angle, tolerance=0.1):
    """
    Return a random (hour, minute) tuple where the angle between hour and minute hands equals target_angle.

    Args:
        target_angle (float): Target angle in degrees (0 ≤ angle ≤ 180).
        tolerance (float): Acceptable error margin in degrees.

    Returns:
        (hour, minute) tuple or None if no such time exists.
    """
    results = times_with_angle(target_angle, tolerance)
    if results:
        return random.choice(results)
    return None


def times_with_angle(target_angle, tolerance=0.1):
    """
    Find all times in a 12-hour period where the angle between hour and minute hands equals target_angle.

    Args:
        target_angle (float): Target angle in degrees (0 ≤ angle ≤ 180).
        tolerance (float): Acceptable error margin in degrees.

    Returns:
        List of (hour, minute) tuples.
    """
    results = []
    for hour in range(12):
        for minute in range(60):
            # Calculate the angle between the hands
            hour_angle = (hour % 12) * 30 + (minute / 60) * 30
            minute_angle = minute * 6
            angle = abs(hour_angle - minute_angle)
            angle = min(angle, 360 - angle)  # smaller angle between hands

            if abs(angle - target_angle) <= tolerance:
                results.append((hour, minute))
    return results
