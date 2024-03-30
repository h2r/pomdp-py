import pomdp_py
import random

description = "testing particle representation"


def test_particles():
    random.seed(100)
    random_dist = {}
    total_prob = 0
    for v in range(4):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]
    for v in random_dist:
        random_dist[v] /= total_prob

    particles = pomdp_py.Particles.from_histogram(
        pomdp_py.Histogram(random_dist), num_particles=int(1e6)
    )

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
    random.seed(100)
    random_dist = {}
    total_prob = 0
    for v in range(5):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]

    particles = pomdp_py.WeightedParticles.from_histogram(random_dist)
    particles_frozen = pomdp_py.WeightedParticles(particles.particles, frozen=True)

    assert particles.hist_valid
    assert particles_frozen.hist_valid

    assert abs(sum(particles[v] for v, _ in particles) - 1.0) <= 1e-6
    assert abs(sum(particles_frozen[v] for v, _ in particles_frozen) - 1.0) <= 1e-6

    for v in random_dist:
        assert abs(particles[v] - random_dist[v] / total_prob) <= 2e-3
        assert abs(particles_frozen[v] - random_dist[v] / total_prob) <= 2e-3

    # Test sampling
    counts = {}
    total = int(1e6)
    for i in range(total):
        v = particles.random()
        counts[v] = counts.get(v, 0) + 1
    for v in counts:
        counts[v] /= total
    for v in random_dist:
        assert abs(counts[v] - random_dist[v] / total_prob) <= 2e-3

    assert particles.mpe() == pomdp_py.Histogram(random_dist).mpe()
    assert particles_frozen.mpe() == pomdp_py.Histogram(random_dist).mpe()
    hash(particles_frozen)

    # Manually add a particle - this changes the distribution
    particles.add(("x1", 0.1))
    assert not particles.hist_valid
    particles.random()  # operation does not re-compute histogram
    assert not particles.hist_valid
    pr_x1 = particles["x1"]  # operation will re-compute histogram
    assert particles.hist_valid

    # However, weights won't sum up to 1 due to the additional particle
    assert abs(sum(particles[v] for v, _ in particles) - 1.0) > 1e-6

    # After condensing, weights sum up to 1
    particles = particles.condense()
    assert abs(sum(particles[v] for v, _ in particles) - 1.0) <= 1e-6

    # Test frozen after condense
    assert not particles.condense().frozen
    assert particles_frozen.condense().frozen


def run():
    test_particles()
    test_weighted_particles()


if __name__ == "__main__":
    run()
