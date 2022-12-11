Event:
  - Creates an event object which calls a specific method after a delay 

EventSimulator:
  - Keeps a queue of Events and calls them step-by-step

Person:
  - Person object has Start time, source, destination
  - Arrive and bus starting methods

Source:
  - Source object has rate of Person generation, # of People, list of queues, and list of people that have arrived
  - creates Person objects and adds to queue

Queue:
  - Creates a queue of People

Bus:
  - Bus object has max capacity, list of queues at stops, and times it takes for the bus to travel
  - Arrive method

Simulation:
  - Simulates Buses, Queues, and People using Events
