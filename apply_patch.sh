# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cp agentdojo.patch agentdojo/
cd agentdojo
git apply agentdojo.patch
rm agentdojo.patch
cd ..

cp torchtune.patch torchtune/
cd torchtune
git apply torchtune.patch
rm torchtune.patch
cd ..
