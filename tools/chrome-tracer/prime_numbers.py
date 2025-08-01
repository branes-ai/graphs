import json
import os
import time

# Re-including the write function from the previous document for self-containment.
def write_viztracker_json(data: dict, file_path: str) -> None:
    """
    Writes a dictionary to a JSON file in a format that can be consumed by Vizviewer.
    The output is formatted for readability.

    Args:
        data: The dictionary containing the data to write.
        file_path: The path to the output JSON file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully wrote data to '{file_path}'.")
    except IOError as e:
        print(f"Error: Could not write to file '{file_path}'. Details: {e}")
        raise

def is_prime(n: int) -> bool:
    """
    Checks if a number is prime using a simple trial division algorithm.
    This is a compute-heavy function, especially for large 'n'.

    Args:
        n: The integer to check.

    Returns:
        True if the number is prime, False otherwise.
    """
    # A prime number is a natural number greater than 1
    if n <= 1:
        return False
    # Check for divisibility from 2 up to the square root of n
    # This optimization reduces the number of checks significantly
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(count: int) -> list:
    """
    Finds a specified number of prime numbers and logs events for each one found.

    Args:
        count: The number of prime numbers to find.

    Returns:
        A list of event dictionaries, one for each prime found.
    """
    prime_events = []
    num_primes_found = 0
    candidate_number = 2

    while num_primes_found < count:
        start_time = time.time()
        
        # This is the "compute-heavy" part we want to measure
        prime_check_start_time = time.time()
        if is_prime(candidate_number):
            prime_check_end_time = time.time()
            
            num_primes_found += 1
            
            event_data = {
                "event_type": "prime_found",
                "prime_number": candidate_number,
                "prime_rank": num_primes_found,
                "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "duration_ms": (prime_check_end_time - prime_check_start_time) * 1000
            }
            prime_events.append(event_data)
            print(f"Event {num_primes_found}: Found prime {candidate_number}")

        candidate_number += 1
    
    return prime_events

# Main program execution
if __name__ == "__main__":
    NUMBER_OF_PRIMES_TO_FIND = 1000  # This will generate 1000 events
    OUTPUT_FILENAME = "prime_finder_viztracker_output.json"

    print(f"Starting to find the first {NUMBER_OF_PRIMES_TO_FIND} prime numbers...")
    
    start_total_time = time.time()
    events = find_primes(NUMBER_OF_PRIMES_TO_FIND)
    end_total_time = time.time()

    total_duration_ms = (end_total_time - start_total_time) * 1000
    print(f"\nFinished in {total_duration_ms:.2f} ms.")
    print(f"Total events generated: {len(events)}")
    
    viztracker_data = {
        "metadata": {
            "source": "PrimeFinder",
            "generation_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "total_duration_ms": total_duration_ms,
            "number_of_primes": NUMBER_OF_PRIMES_TO_FIND
        },
        "events": events
    }
    
    # Write the collected events to a JSON file
    write_viztracker_json(viztracker_data, OUTPUT_FILENAME)
