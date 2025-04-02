def levenshtein_distance(str1, str2):
    i, j = 0, 0  # Initialize pointers for both strings
    steps = []  # To keep track of the steps

    while i < len(str1) and j < len(str2):
        if str1[i] == str2[j]:
            steps.append(f"No operation needed: '{str1[i]}' matches '{str2[j]}'")
            i += 1
            j += 1
        else:
            steps.append(f"Substitution: '{str1[i]}' -> '{str2[j]}'")
            i += 1
            j += 1

    # If there are remaining characters in str1, it means we need deletions
    while i < len(str1):
        steps.append(f"Deletion: '{str1[i]}' removed")
        i += 1

    # If there are remaining characters in str2, it means we need insertions
    while j < len(str2):
        steps.append(f"Insertion: '{str2[j]}' added")
        j += 1

    # The distance is the number of operations we've done
    return len(steps), steps


# Take user input for two strings
str1 = input("Enter the first string: ")
str2 = input("Enter the second string: ")

# Calculate the Levenshtein distance and get the steps
distance, steps = levenshtein_distance(str1, str2)

# Print the steps and the final result
print(f"\nSteps to transform '{str1}' to '{str2}':\n")
for step in steps:
    print(step)

print(f"\nThe Levenshtein distance between '{str1}' and '{str2}' is: {distance}")
