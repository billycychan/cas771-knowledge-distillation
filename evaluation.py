import os
import sys
import subprocess
import time
import traceback
import platform

def run_evaluation_script(script_name):
    """
    Run an evaluation script and handle any exceptions
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    try:
        # Run the script using Python's subprocess module
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n{'-'*80}")
        print(f"{script_name} completed successfully!")
        print(f"{'-'*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'-'*80}")
        print(f"Error running {script_name}!")
        print(f"Return code: {e.returncode}")
        print(f"{'-'*80}\n")
        return False
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"Exception while running {script_name}:")
        print(str(e))
        traceback.print_exc()
        print(f"{'-'*80}\n")
        return False

def main():
    """
    Main function to run all evaluation scripts
    """
    print("\nKnowledge Distillation Evaluation Suite")
    print("======================================\n")
    
    # Store current directory to ensure correct path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get paths to evaluation scripts
    taska_script = os.path.join(current_dir, "evaluation_taska.py")
    taskb_script = os.path.join(current_dir, "evaluation_taskb.py")
    
    # Check if scripts exist
    if not os.path.exists(taska_script):
        print(f"Error: {taska_script} not found!")
        return
    
    if not os.path.exists(taskb_script):
        print(f"Error: {taskb_script} not found!")
        return
    
    # Run TaskA evaluation
    taska_success = run_evaluation_script(taska_script)
    
    # Run TaskB evaluation
    taskb_success = run_evaluation_script(taskb_script)
    
    # Print summary
    print("\nEvaluation Summary")
    print("=================")
    print(f"TaskA Evaluation: {'Successful' if taska_success else 'Failed'}")
    print(f"TaskB Evaluation: {'Successful' if taskb_success else 'Failed'}")
    print("\nSee the generated output directories for detailed results and visualizations.")
    
    # Display sample prediction images
    if taska_success or taskb_success:
        print("\nDisplaying sample prediction images...")
        
        # Paths to the sample prediction images
        taska_image = os.path.join(current_dir, "evaluation", "TaskA", "sample_predictions.png")
        taskb_image = os.path.join(current_dir, "evaluation", "TaskB", "sample_predictions.png")
        
        # Check if the images exist
        if taska_success and os.path.exists(taska_image):
            print(f"Opening TaskA sample predictions: {taska_image}")
            # Use the appropriate command based on the operating system
            if platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", taska_image])
            elif platform.system() == "Windows":
                os.startfile(taska_image)
            else:  # Linux
                subprocess.Popen(["xdg-open", taska_image])
        
        if taskb_success and os.path.exists(taskb_image):
            print(f"Opening TaskB sample predictions: {taskb_image}")
            # Use the appropriate command based on the operating system
            if platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", taskb_image])
            elif platform.system() == "Windows":
                os.startfile(taskb_image)
            else:  # Linux
                subprocess.Popen(["xdg-open", taskb_image])

if __name__ == "__main__":
    main()
