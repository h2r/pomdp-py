import pomdp_py
import random

def test_particles():
    random_dist = {}
    total_prob = 0
    for v in range(20):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]
    for v in random_dist:
        random_dist[v] /= total_prob

    particles = pomdp_py.Particles.from_histogram(pomdp_py.Histogram(random_dist), num_particles=int(1e6))

    for v in random_dist:
        print(random_dist[v], particles[v])
        assert abs(particles[v] - random_dist[v]) <= 1e-3

    counts = {}
    total = int(1e6)
    for i in range(total):
        v = particles.random()
        counts[v] = counts.get(v, 0) + 1
    for v in counts:
        counts[v] /= total
    for v in random_dist:
        assert abs(counts[v] - random_dist[v]) <= 1e-3

    assert particles.mpe() == pomdp_py.Histogram(random_dist).mpe()


def test_weighted_particles():
    random_dist = {}
    total_prob = 0
    for v in range(5):
        random_dist[f"x{v}"] = random.uniform(0, 1)
        total_prob += random_dist[f"x{v}"]
    for v in random_dist:
        random_dist[v] /= total_prob
    print(sum(random_dist[v] for v in random_dist))

    particles = pomdp_py.WeightedParticles.from_histogram(pomdp_py.Histogram(random_dist))
    print(sum(random_dist[v] for v in random_dist))

    assert sum(particles[v] for v, _ in particles) == 1.0

    for v in random_dist:
        print(random_dist[v], particles[v])
        assert abs(particles[v] - random_dist[v]) <= 1e-3

    counts = {}
    total = int(1e6)
    for i in range(total):
        v = particles.random()
        counts[v] = counts.get(v, 0) + 1
    for v in counts:
        counts[v] /= total
    for v in random_dist:
        assert abs(counts[v] - random_dist[v]) <= 1e-3

    assert particles.mpe() == pomdp_py.Histogram(random_dist).mpe()

if __name__ == "__main__":
    test_particles()
    test_weighted_particles()
