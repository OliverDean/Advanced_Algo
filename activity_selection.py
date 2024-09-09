def activity_selection(activities):
    """
    This function selects the maximum number of compatible activities from a list of intervals using a greedy approach.

    Problem Overview:
    - You are given a set of activities where each activity is represented as a time interval [start, finish).
    - The goal is to select the largest subset of activities such that no two activities overlap.
    - Activities are compatible if they do not overlap in time.

    Greedy Strategy:
    - Sort the activities by their finish time.
    - Greedily pick the activity that finishes the earliest and discard any overlapping activities.
    - Repeat this until all non-overlapping activities are selected.

    Time Complexity:
    - Sorting the activities takes O(N log N), where N is the number of activities.
    - The selection process takes O(N), resulting in O(N log N) overall.

    Space Complexity:
    - O(N) for storing the result list.

    Parameters:
    - activities: A list of tuples, where each tuple represents an activity (start time, finish time).

    Returns:
    - A list of non-overlapping activities that can be selected.
    """
    
    # Sort activities by their finish time
    activities.sort(key=lambda a: a[1])

    result = []  # Store the selected activities
    t = 0  # Track the time of the last selected activity's finish time

    for activity in activities:
        start, finish = activity
        if start >= t:  # If the current activity starts after or when the last one finishes
            result.append(activity)
            t = finish  # Update the time to the finish of the current activity

    return result

# Example usage
activities = [(6, 9), (1, 10), (2, 4), (1, 7), (5, 6), (8, 11), (9, 11)]
selected_activities = activity_selection(activities)
print(selected_activities)
