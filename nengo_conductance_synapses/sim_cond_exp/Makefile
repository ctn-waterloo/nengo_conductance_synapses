#   This file is part of soft_cond_lif
#   (c) Andreas Stöckel 2017
#
#   soft_cond_lif is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   soft_cond_lif is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with soft_cond_lif.  If not, see <http://www.gnu.org/licenses/>.

all: /tmp/sim_cond_exp_pipe

/tmp/sim_cond_exp_pipe: sim_cond_exp_pipe.cpp
	c++ -std=c++14 -O3 sim_cond_exp_pipe.cpp -o /tmp/sim_cond_exp_pipe -ffast-math -fno-finite-math-only -pthread

