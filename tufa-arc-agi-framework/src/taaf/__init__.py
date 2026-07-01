"""ARC-AGI3 orchestration framework (TAAF) — Tufa Labs.

No top-level re-exports — types live on their submodules
(``taaf.game.GameState``, not ``taaf.GameState``).
"""

# ``as <name>`` aliases plus ``__all__`` mark these as intentional
# re-exports for pyright strict (otherwise: reportUnusedImport on each line).
import taaf.benchmark as benchmark
import taaf.competition_arcade as competition_arcade
import taaf.deploy as deploy
import taaf.deploy_inline as deploy_inline
import taaf.deploy_kaggle as deploy_kaggle
import taaf.deploy_slurm as deploy_slurm
import taaf.diagnostics as diagnostics
import taaf.game as game
import taaf.game_api as game_api
import taaf.game_examples as game_examples
import taaf.solver as solver
import taaf.solver_examples as solver_examples
import taaf.standard_benchmarks as standard_benchmarks
import taaf.support as support

__all__ = [
    "benchmark",
    "competition_arcade",
    "deploy",
    "deploy_inline",
    "deploy_kaggle",
    "deploy_slurm",
    "diagnostics",
    "game",
    "game_api",
    "game_examples",
    "solver",
    "solver_examples",
    "standard_benchmarks",
    "support",
]
