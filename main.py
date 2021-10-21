"""
Entry point script to run the tasks.
"""
import argparse
import time

from src import tasks


def main():
    parser = argparse.ArgumentParser(description='data-analysis-tasks')
    parser.add_argument("-tl", "--taskslist", default='all', 
                        help='comma separated list of task names')
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="print output on console")
    
    args = parser.parse_args()
    verbose = args.verbose
    output = None
    for idx, task_name in enumerate(args.taskslist.split(',')):
        if verbose:
            start = time.perf_counter()
            print('{}. Running {} task...'.format(idx+1, task_name))

        task = getattr(tasks, task_name)()
        output = task.run(output, verbose=verbose)

        if verbose:
            end = time.perf_counter()
            print('{} task has completed. Total time taken {} secs.\n'.format(
                task_name, end-start))
          
          
if __name__ == '__main__':
    main()
