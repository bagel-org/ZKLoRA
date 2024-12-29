import merkle
print(merkle.add(3, 5)) 
print(merkle.minus(3, 5)) 

import random

# Create a list of 10 random floating point numbers
random_numbers = [random.uniform(-100, 100) for _ in range(10)]
print("Random numbers:", random_numbers)

# Get the Merkle root hash of these numbers
merkle_root = merkle.insert_values(random_numbers)
print("Merkle root:", merkle_root)
