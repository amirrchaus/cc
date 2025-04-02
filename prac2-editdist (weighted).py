def weighted_edit_distance(s1, s2, insert_cost=1, delete_cost=1, substitute_cost=2):
    m, n = len(s1), len(s2)
    
    # Create a matrix to store the costs
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i * delete_cost  # Cost of deleting all characters from s1
    for j in range(n + 1):
        dp[0][j] = j * insert_cost  # Cost of inserting all characters into s1
    
    # Fill the DP table with the minimum cost at each position
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if characters are the same
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + delete_cost,  # Deletion
                    dp[i][j - 1] + insert_cost,  # Insertion
                    dp[i - 1][j - 1] + substitute_cost  # Substitution
                )
    
    # Backtrack to find the operations performed
    steps = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # No operation needed
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + delete_cost):
            # Deletion
            steps.append(f"Delete '{s1[i - 1]}' from s1")
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + insert_cost):
            # Insertion
            steps.append(f"Insert '{s2[j - 1]}' into s1")
            j -= 1
        else:
            # Substitution
            steps.append(f"Substitute '{s1[i - 1]}' with '{s2[j - 1]}'")
            i -= 1
            j -= 1
    
    # The operations are collected in reverse, so reverse them before returning
    steps.reverse()
    
    return dp[m][n], steps

# Example usage:
s1 = input("Enter String 1: ")
s2 = input("Enter String 2: ")

distance, steps = weighted_edit_distance(s1, s2, insert_cost=1, delete_cost=1, substitute_cost=2)

print(f"Weighted edit distance between '{s1}' and '{s2}': {distance}")
print("\nSteps to transform the strings:")
for step in steps:
    print(step)
