## Takeaways from the Tufa Lab Solution

- The strongest direction appears to be combining a capable harness with reinforcement learning.
- Build a dashboard that can show agent runs live, replay completed attempts, and make it easy to inspect the reasoning process.
- Lean into the model's natural biases instead of fighting them. A good harness should expose where the model is already strong and help amplify those behaviors.
- Base model selection matters a lot. Larger models are generally better, but post-training can matter more than raw size. Tufa Lab used Qwen 3.6 27B, which reportedly outperformed both Qwen 3.5 and Qwen 3.6 35B. That suggests something about the 27B model's post-training made it unusually well suited to the task.
- After harness engineering, post-training with RL may be the next major source of gains.
- Reward shaping may be useful for RL, especially if the reward can capture intermediate progress rather than only final task success.
- Use the model's default reasoning settings unless there is evidence that changing them improves performance.
- Frontier models tend to be more observant and careful before making a move.
- Smaller models seem to have a shorter effective reasoning span. They struggle to connect information across distant turns, and cutting context does not appear to hurt much. A 64k context window is probably enough.

## Notes on the Duck Harness

- The harness should support both image and ASCII representations.
- The task can be framed as a coding problem, which may make it easier for models to reason about state, actions, and transformations.