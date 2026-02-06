# RL-based Task Graph Optimization for Taskflow
import numpy as np
import tensorflow as tf
from taskflow.task import Task
from taskflow.workflow import Workflow

class RLTaskGraphOptimizer:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def optimize(self, workflow):
        tasks = workflow.tasks()
        state = self._get_state(tasks)
        action = self.model.predict(state)[0][0]
        new_workflow = self._apply_action(workflow, action)
        return new_workflow

    def _get_state(self, tasks):
        # Example state: number of tasks and their dependencies
        return np.array([[len(tasks), len([t for t in tasks if t.dependents])]], dtype=float)

    def _apply_action(self, workflow, action):
        # Example action: add a new task with a random dependency
        new_task = Task(name=f'new_task_{len(workflow.tasks())}', func=lambda: None)
        if len(workflow.tasks()) > 0:
            new_task.add_dependency(np.random.choice(workflow.tasks()))
        workflow.add_task(new_task)
        return workflow

# Example usage
if __name__ == '__main__':
    wf = Workflow()
    wf.add_task(Task(name='task1', func=lambda: None))
    wf.add_task(Task(name='task2', func=lambda: None, depends_on=['task1']))
    optimizer = RLTaskGraphOptimizer()
    optimized_wf = optimizer.optimize(wf)
    print(optimized_wf)