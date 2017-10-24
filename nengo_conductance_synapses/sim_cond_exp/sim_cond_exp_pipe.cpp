/*
 *  This file is part of soft_cond_lif
 *
 *  soft_cond_lif is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  soft_cond_lif is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with soft_cond_lif.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <atomic>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cinder/cinder.hpp>

#include "json.hpp"

using namespace cinder;
using json = nlohmann::json;

/**
 * Uses RTTI information to convert a JSON object to a Cinder typed vector\
 * structure.
 */
template <typename T>
T json_to_typed_vector(const json &j)
{
	constexpr auto names = T::names();
	T res;
	for (size_t i = 0; i < T::size(); i++) {
		if (j.find(names[i]) != j.end()) {
			res[i] = j[names[i]];
		}
	}
	return res;
}

/**
 * Structure holding all information for a single experiment run.
 */
struct Task {
	std::string id;
	Time T;
	Real rate_e;
	Real rate_i;
	size_t repeat;
	size_t seed;
	Real spike_loss_e;
	Real spike_loss_i;
	LIFParameters lif_params;
	CondExpParameters syn_e_params;
	CondExpParameters syn_i_params;
};

int main()
{
	std::queue<Task> queue;
	std::mutex queue_mutex, io_mutex;
	std::atomic<bool> cancel(false);

	auto f = [&]() {
		while (true) {
			Task current_task;

			// Fetch a task
			while (true) {
				{
					std::lock_guard<std::mutex> lock(queue_mutex);
					if (!queue.empty()) {
						current_task = queue.front();
						queue.pop();
						break;
					}
				}
				if (cancel) {
					return;  // No task available, we're done
				}
				// Wait a short while for another task
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(10ms);
			}

			// Some shorthands
			const Real r_e = current_task.rate_e;
			const Real r_i = current_task.rate_i;
			const size_t seed = current_task.seed;
			const Time T = current_task.T;

			std::vector<Real> results;

			for (size_t j = 0; j < current_task.repeat; j++) {
				// Generate input spike trains
				auto spikes_e = simulate_spike_loss(
				    poisson_spike_train(0_s, T, std::abs(r_e),
				                        seed + 2 * j + 0),
				    current_task.spike_loss_e, seed + 2 * j + 0);
				auto spikes_i = simulate_spike_loss(
				    poisson_spike_train(0_s, T, std::abs(r_i),
				                        seed + 2 * j + 1),
				    current_task.spike_loss_i, seed + 2 * j + 1);

				// Negate the weight for negative rates
				CondExpParameters sep = current_task.syn_e_params;
				CondExpParameters sip = current_task.syn_i_params;
				sep[CondExpParameters::idx_w_syn] *= (r_e > 0.0) ? 1.0 : -1.0;
				sip[CondExpParameters::idx_w_syn] *= (r_i > 0.0) ? 1.0 : -1.0;

				EulerIntegrator integrator;
				NullRecorder recorder;
				NullController controller;
				auto current_source = make_current_source(
				    CondExp(sep, spikes_e), CondExp(sip, spikes_i));

				size_t spike_count = 0;
				auto neuron = make_neuron<LIF>(current_source,
				                               [&spike_count, T](Time t) {
					                               if (t > T * 0.5) {
						                               spike_count++;
					                               }
					                           },
				                               current_task.lif_params);

				make_solver(neuron, integrator, recorder, controller)
				    .solve(T, 0.1_ms);

				results.emplace_back(2.0 * Real(spike_count) / T.sec());
			}

			// Write the result to std::cout
			{
				std::lock_guard<std::mutex> lock(io_mutex);
				std::cout << current_task.id;
				for (Real res : results) {
					std::cout << "," << res;
				}
				std::cout << std::endl;
			}
		}
	};

	// Spawn n threads
	std::vector<std::thread> threads;
	for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
		threads.emplace_back(std::thread(f));
	}

	std::string line;
	while (std::getline(std::cin, line)) {
		// Skip empty lines
		if (line.empty()) {
			continue;
		}

		// Parse the given line into a JSON datastructure
		json j = json::parse(line);

		// Read the task from the JSON object
		Task t;
		t.id = j["input"]["id"];
		t.T = Time::sec(j["input"]["T"]);
		t.rate_e = j["input"]["rate_e"];
		t.rate_i = j["input"]["rate_i"];
		t.repeat = j["input"]["repeat"];
		t.seed = j["input"]["seed"];
		t.spike_loss_e = j["input"]["spike_loss_e"];
		t.spike_loss_i = j["input"]["spike_loss_i"];
		t.lif_params = json_to_typed_vector<LIFParameters>(j["neuron"]);
		t.syn_e_params = json_to_typed_vector<CondExpParameters>(j["syn_e"]);
		t.syn_i_params = json_to_typed_vector<CondExpParameters>(j["syn_i"]);

		// Emplace the task on the task queue
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			queue.emplace(t);
		}
	}

	// Wait for all threads to finish
	cancel = true;
	for (size_t i = 0; i < threads.size(); i++) {
		threads[i].join();
	}

	return 0;
}

