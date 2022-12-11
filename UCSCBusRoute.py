
from collections import deque # Used to implement queues.
import random # Random choice, etc.
import heapq # Used in discrete event simulator
import numpy as np # Used for gamma probability distribution, and percentiles.
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate # To display the bus status.

def fmt(x):
    """Formats a number x which can be None, for convenience."""
    return None if x is None else "{:.2f}".format(x)

class Event(object):

    def __init__(self, method, delay=0, args=None, kwargs=None):
        """An event consists in calling a specified method after a delay,
        with a given list of args and kwargs."""
        self.method = method
        self.delay = delay
        self.args = args or []
        self.kwargs = kwargs or {}
        self.time = None # Not known until this is added to the queue.

    def __call__(self, time):
        """This causes the event to happen, returning a list of new events
        as a result. Informs the object of the time of occurrence."""
        return self.method(*self.args, time=time, **self.kwargs)

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return "@{}: {} {} {} {} dt={:.2f}".format(
            fmt(self.time),
            self.method.__self__.__class__.__name__,
            self.method.__name__,
            self.args, self.kwargs, self.delay
        )

class EventSimulator(object):

    def __init__(self, trace=False):
        self.events = []
        self.time = 0 # This is the global time.
        self.trace = trace

    def add_event(self, event):
        """Adds an event to the queue."""
        event.time = self.time + event.delay
        heapq.heappush(self.events, event)

    def step(self):
        """Performs a step of the simulation."""
        if len(self.events) > 0:
            event = heapq.heappop(self.events)
            self.time = event.time
            new_events = event(self.time) or []
            for e in new_events:
                self.add_event(e)
            if self.trace:
                print("Processing:", event)
                print("New events:", new_events)
                print("Future events:", self.events)

    def steps(self, number=None):
        """Performs at most number steps (or infinity, if number is None)
        in the simulation."""
        num_done = 0
        while len(self.events) > 0:
            self.step()
            num_done += 1
            if num_done == number:
                break



class Person(object):

    def __init__(self, start_time, source, destination, have_arrived,
                 person_id=None):
        """
        @param start_time: time at which a person enters the system.
        @param source: stop at which the person wants to climb on.
        @param destination: destination stop.
        @param have_arrived: list of people who have arrived, so we can
            plot their bus time.
        """
        self.start_time = start_time
        self.bus_time = None # Time at which climbed on bus
        self.end_time = None
        self.source = source
        self.destination = destination
        self.have_arrived = have_arrived
        # for id purpose
        self.id = person_id

    # Event method
    def arrived(self, time=None):
        """The person has arrived to their destination."""
        self.end_time = time
        self.have_arrived.append(self)
        return [] # No events generated as a consequence.

    def start_bus(self, time=None):
        """The person starts getting on the bus."""
        self.bus_time = time

    @property
    def elapsed_time(self):
        return None if self.end_time is None else self.end_time - self.start_time

    @property
    def travel_time(self):
        return None if self.end_time is None else self.end_time - self.bus_time

    @property
    def wait_time(self):
        return None if self.end_time is None else self.bus_time - self.start_time

    def __repr__(self):
        return f"Person #: {self.id}, source: {self.source}, dest: {self.destination}"



class Source(object):
    """Creates people, and adds them to the queues."""

    def __init__(self, rate=1., queue_ring=None, number=None, have_arrived=None):
        """
        @param rate is the rate at which people are generated.
        @param number is the total number of people to generate; None = unlimited.
        @param queue_ring is the queue ring (a list of queues) where people are added.
        @param have_arrived is the list where people who have arrived are added.
        """
        self.rate = rate
        self.queue_ring = queue_ring
        self.num_stops = len(queue_ring)
        self.number = number
        self.have_arrived = have_arrived
        self.person_id = 0 # For debugging.

    # Event method
    def start(self, time=None):
        if self.number == 0:
            return [] # Nothing more to be done.
        # Creates the person
        self.person_id += 1
        source, destination = random.sample(range(self.num_stops), 2)
        person = Person(time, source, destination, self.have_arrived,
                        person_id = self.person_id)
        queue = self.queue_ring[source]
        enter_event = Event(queue.enter, args=[person])
        # Schedules the next person creation.
        self.number = None if self.number is None else self.number - 1
        dt = np.random.gamma(1, 1/self.rate)
        start_event = Event(self.start, delay=dt)
        return [enter_event, start_event]


### Class Queue

class Queue(object):

    def __init__(self):
        """We create a queue."""
        # YOUR CODE HERE
        self.num_ppl = 0
        self.person_list = []
    # Event method
    def enter(self, person, time=None):
        # YOUR CODE HERE
        self.num_ppl += 1
        self.person_list.append(person)
        return Event(self.enter, delay = 0)

    ### You can put here any other methods that might help you.
    # YOUR CODE HERE
    def get_all(self):
        in_queue = self.num_ppl
        self.num_ppl = 0
        return in_queue
    def get_list(self):
        return self.person_list



#@ title Class Bus

class Bus(object):

    def __init__(self, queue_ring, max_capacity, geton_time, nextstop_time,
                 bus_id=None):
        """The bus is created with the following parameters:
        @param max_capacity: the max capacity of the bus.
        @param queue_ring: the ring (list) of queues representing the stops.
        @param geton_time: the expected time that it takes for a person to climb
            the 2 steps to get on the bus.  The time a person takes to get on is
            given by np.random.gamma(2, geton_time / 2).
            This is the same as the time to get off the bus.
        @param nextstop_time: the average time the bus takes to go from one stop
            to the next.  The actual time is given by
            np.random.gamma(10, nextstop_time/10).
        @param bus_id: An id for the bus, for debugging.
        """
        self.queue_ring = queue_ring
        self.max_capacity = max_capacity
        self.geton_time = geton_time
        self.nextstop_time = nextstop_time
        self.id = bus_id
        ### Put any other thing you need in the initialization below.
        # YOUR CODE HERE
        self.bus_ppl = 0
    @property
    def stop(self):
        """Returns the current (most recent) stop of the bus,
        as an integer."""
        # YOUR CODE HERE
        return 
    @property
    def onboard(self):
        """Returns the list of people on the bus."""
        # YOUR CODE HERE
        
    @property
    def occupancy(self):
        """Returns the number of passengers on the bus."""
        # YOUR CODE HERE

    # Event method.
    def arrive(self, stop_idx, time=None):
        """Arrives at the next stop."""
        ### You can do what you want here.
        # YOUR CODE HERE
        self.stop_idx = stop_idx
        return Event(self.arrive, delay = np.random.gamma(2, geton_time / 2))
    
    def __repr__(self):
        """This will print a bus, which helps in debugging."""
        return "Bus#: {}, #people: {}, dest: {}".format(
            self.id, self.occupancy, [p.destination for p in self.onboard])

    ### You can have as many other methods as you like, including other
    ### events for the bus.  Up to you.
    # YOUR CODE HERE



def bus_distance(ix, iy, num_stops=20):
    """Returns the distance between two buses."""
    if ix is None or iy is None:
        return None
    d1 = (ix - iy + num_stops) % num_stops
    d2 = (iy - ix + num_stops) % num_stops
    return min(d1, d2)


class Simulation(object):

    def __init__(self, num_stops=20, num_buses=1,
                 bus_nextstop_time=1, bus_geton_time=0.1,
                 bus_max_capacity=50,
                 person_rate=2, destinations="random",
                 number_of_people=None,
                 trace=False):
        self.num_stops = num_stops
        self.num_buses = num_buses
        self.bus_max_capacity = bus_max_capacity
        # Chooses the initial stops for the buses.
        self.initial_stops = list(np.mod(np.arange(0, self.num_buses) * max(1, num_stops // num_buses), num_stops))
        # Speeds
        self.bus_nextstop_time = bus_nextstop_time
        self.bus_geton_time = bus_geton_time
        self.person_rate = person_rate
        # Event simulator
        self.simulator = EventSimulator(trace=trace)
        # Builds the queue ring
        self.queue_ring = [Queue() for _ in range(num_stops)]
        # And the source.
        self.have_arrived = []
        self.source = Source(rate=person_rate, queue_ring=self.queue_ring,
                             number=number_of_people, have_arrived=self.have_arrived)
        # And the buses.
        self.buses = [Bus(queue_ring=self.queue_ring,
                          max_capacity=bus_max_capacity,
                          geton_time=bus_geton_time,
                          nextstop_time=bus_nextstop_time,
                          bus_id=i + 1)
            for i in range(num_buses)]
        # We keep track of the distances between buses, and the
        # bus occupancies.
        self.positions = [[] for _ in range(num_buses)]
        self.occupancies = [[] for _ in range(num_buses)]


    def start(self):
        """Starts the simulation."""
        # Injects the initial events in the simulator.
        # Source.
        self.simulator.add_event(Event(self.source.start))
        # Buses.
        for i, bus in enumerate(self.buses):
            self.simulator.add_event(
                Event(bus.arrive, args=[self.initial_stops[i]]))

    def step(self):
        """Performs a step in the simulation."""
        self.simulator.step()
        for bus_idx in range(self.num_buses):
            self.positions[bus_idx].append(self.buses[bus_idx].stop)
            self.occupancies[bus_idx].append(self.buses[bus_idx].occupancy)

    def plot(self):
        """Plots the history of positions and occupancies."""
        # Plots positions.
        for bus_idx in range(self.num_buses):
            plt.plot(self.positions[bus_idx])
        plt.title("Positions")
        plt.show()
        # Plots occupancies.
        for bus_idx in range(self.num_buses):
            plt.plot(self.occupancies[bus_idx])
        plt.title("Occupancies")
        plt.show()
        # Plots times.
        plt.hist([p.wait_time for p in self.have_arrived])
        plt.title("Wait time")
        plt.show()
        plt.hist([p.travel_time for p in self.have_arrived])
        plt.title("Time on the bus")
        plt.show()
        plt.hist([p.elapsed_time for p in self.have_arrived])
        plt.title("Total time")
        plt.show()
        # Plots bus distances
        if self.num_buses > 1:
            for i, j in itertools.combinations(range(self.num_buses), 2):
                ds = [bus_distance(pi, pj, num_stops=self.num_stops)
                      for pi, pj in zip(self.positions[i], self.positions[j])]
                plt.plot(ds)
            plt.title("Bus distances")
            plt.show()

    def status(self):
        """Tabulates the bus location and queue status."""
        headers = ["Stop Index", "Queue", "Buses"]
        rows = []
        for stop_idx, queue in enumerate(simulation.queue_ring):
            buses = [b for b in self.buses if b.current_stop == stop_idx]
            busStr = "\n".join([bus.__str__() for bus in buses])
            personStr = "\n".join([person.__str__() for person in queue.people])
            row = [f"{stop_idx}", f"{personStr}", f"{busStr}"]
            rows.append(row)
        print(tabulate(rows, headers, tablefmt="grid", stralign='left', numalign='right'))

"""Here is how to use the above `status` mtethod to debug your application. """

simulation = Simulation(num_stops=5, num_buses=2, person_rate=2, trace=False)
simulation.start()
for i in range(30):
    simulation.step()
    print(f"\nState after step {i}")
    simulation.status()



sim = Simulation(trace=False, person_rate=2,
                 num_stops=12,
                 bus_nextstop_time=2, bus_max_capacity=45, num_buses=1)

sim.start()

for _ in range(5000):
    sim.step()

sim.plot()


def check_all_stops(locations, num_stops):
    """The bus stops at all stops"""
    for i in range(len(locations) - 1):
        if locations[i] is not None:
            assert (locations[i] == locations[i + 1] or
                    (locations[i] + 1) % num_stops == locations[i + 1]), (locations[i], locations[i + 1])

def check_occupancies(sim):
    for bus_idx in range(sim.num_buses):
        for oc in sim.occupancies[bus_idx]:
            assert 0 <= oc <= sim.bus_max_capacity

def check_no_ghosts(locations):
    """The location is always defined, except possibly at the beginning."""
    for i in range(len(locations) - 1):
        if locations[i] is not None:
            assert locations[i + 1] is not None

def count_tours(locations):
    n = 0
    for i in range(len(locations) - 1):
        if locations[i] == 0 and locations[i + 1] == 1:
            n += 1
    return n

def check_simulation(sim):
    sim.start()
    for _ in range(10000):
        sim.step()
    simulation_time = sim.simulator.time
    min_tours = sim.simulator.time / (4 * sim.num_stops *
                                    (sim.bus_nextstop_time + 10 * sim.bus_geton_time))

    # Checks the bus tours.
    for bus_idx in range(sim.num_buses):
        check_all_stops(sim.positions[bus_idx], sim.num_stops)
        check_no_ghosts(sim.positions[bus_idx])
        assert count_tours(sim.positions[bus_idx]) > min_tours
    # Checks occupancies.
    check_occupancies(sim)



sim = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=1)

check_simulation(sim)

travel_times = [p.travel_time for p in sim.have_arrived]
total_times = [p.elapsed_time for p in sim.have_arrived]
wait_times = [p.wait_time for p in sim.have_arrived]
print("Travel:", np.average(travel_times), np.quantile(travel_times, 0.90))
print("Total:", np.average(total_times), np.quantile(total_times, 0.90))
print("Wait:", np.average(wait_times), np.quantile(wait_times, 0.90))
assert 15 < np.average(travel_times) < 24
assert 30 < np.quantile(travel_times, 0.9) < 40
assert 35 < np.average(total_times) < 48
assert 50 < np.quantile(total_times, 0.9) < 75
assert 18 < np.average(wait_times) < 25
assert 30 < np.quantile(wait_times, 0.9) < 48



sim = Simulation(trace=False, person_rate=2,
                 num_stops=12,
                 bus_nextstop_time=2, bus_max_capacity=50, num_buses=2)

sim.start()

for _ in range(5000):
    sim.step()

sim.plot()




sim1 = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=1)

sim2 = Simulation(trace=False, person_rate=2,
            num_stops=12,
            bus_nextstop_time=2, bus_max_capacity=50, num_buses=2)

sim1.start()
for _ in range(10000):
    sim1.step()

check_simulation(sim2)