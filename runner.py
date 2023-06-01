import threading
import subprocess

def run_script():
    subprocess.call(['python', './generate/simulate_errorsV2.py'])

# Create threads
threads = []
for i in range(5):  # replace 5 with the number of threads you want
    t = threading.Thread(target=run_script)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()