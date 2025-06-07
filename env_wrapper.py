# ----  Frame Skipping Wrapper atari----
class FrameSkipEnv:
    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip
        self.lives=0
    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale
    def reset(self):
        # NoopReset
        _, reset_info = self.env.reset()

        noops = self.env.unwrapped.np_random.integers(1, 31)

        for _ in range(noops):
            state, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                state, reset_info = self.env.reset()
        self.lives = self.ale.lives()
        return state,reset_info

    def render(self, render_mode):
        return self.env.render(render_mode=render_mode)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
# 2013Xiv--the environment is considered to have terminated whenever a life is lost----对于有几条命的环境如果掉命设为终止有助于加速收敛
            new_lives = self.ale.lives()
            terminated = terminated or new_lives < self.lives
            self.game_over = terminated
            self.lives = new_lives
            done = terminated or truncated
            if done:
                break
        return state, total_reward, done, info