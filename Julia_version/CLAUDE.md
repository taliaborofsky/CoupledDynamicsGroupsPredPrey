# Editing guidelines

- Make only the minimal edits required to fix the problem at hand. Do not rewrite, restructure, or clean up surrounding code unless explicitly asked.
- When fixing a bug in a notebook cell, preserve all existing logic in that cell and change only what is broken.
- Do not rename variables, reorder arguments, or refactor functions as a side effect of a bug fix.
- When adding a new function to any submodule (GroupsOnly.jl, ModelFuns.jl, etc.), also add the corresponding `export` line in CooperativeHuntingPkg/src/CooperativeHuntingPkg.jl.
