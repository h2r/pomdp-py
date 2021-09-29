import pomdp_py
import random

description = "testing particle representation"

def test_particles():
    random_dist = {}
    total_prob = 0
    for v in range(4):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]
    for v in random_dist:
        random_dist[v] /= total_prob

    particles = pomdp_py.Particles.from_histogram(pomdp_py.Histogram(random_dist), num_particles=int(1e6))

    for v in random_dist:
        assert abs(particles[v] - random_dist[v]) <= 2e-3

    counts = {}
    total = int(1e6)
    for i in range(total):
        v = particles.random()
        counts[v] = counts.get(v, 0) + 1

    for v in counts:
        counts[v] /= total

    for v in random_dist:
        assert abs(counts[v] - random_dist[v]) <= 2e-3

    assert particles.mpe() == pomdp_py.Histogram(random_dist).mpe()


def test_weighted_particles():
    random_dist = {}
    total_prob = 0
    for v in range(5):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]
    for v in random_dist:
        random_dist[v] /= total_prob

    particles = pomdp_py.WeightedParticles.from_histogram(pomdp_py.Histogram(random_dist))

    assert abs(sum(particles[v] for v, _ in particles) - 1.0) <= 1e-6

    for v in random_dist:
        assert abs(particles[v] - random_dist[v]) <= 2e-3

    counts = {}
    total = int(1e6)
    for i in range(total):
        v = particles.random()
        counts[v] = counts.get(v, 0) + 1
    for v in counts:
        counts[v] /= total
    for v in random_dist:
        assert abs(counts[v] - random_dist[v]) <= 2e-3

    assert particles.mpe() == pomdp_py.Histogram(random_dist).mpe()

def run():
    test_particles()
    test_weighted_particles()


if __name__ == "__main__":
    run()
