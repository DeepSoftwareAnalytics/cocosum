public Event next(int minIp, int maxIp) {
		final double p = rnd.nextDouble();

		if (p * 1000 >= states.size()) {
			// create a new state machine
			final int nextIP = rnd.nextInt(maxIp - minIp) + minIp;

			if (!states.containsKey(nextIP)) {
				EventTypeAndState eventAndState = State.Initial.randomTransition(rnd);
				states.put(nextIP, eventAndState.state);
				return new Event(eventAndState.eventType, nextIP);
			}
			else {
				// collision on IP address, try again
				return next(minIp, maxIp);
			}
		}
		else {
			// pick an existing state machine

			// skip over some elements in the linked map, then take the next
			// update it, and insert it at the end

			int numToSkip = Math.min(20, rnd.nextInt(states.size()));
			Iterator<Entry<Integer, State>> iter = states.entrySet().iterator();

			for (int i = numToSkip; i > 0; --i) {
				iter.next();
			}

			Entry<Integer, State> entry = iter.next();
			State currentState = entry.getValue();
			int address = entry.getKey();

			iter.remove();

			if (p < errorProb) {
				EventType event = currentState.randomInvalidTransition(rnd);
				return new Event(event, address);
			}
			else {
				EventTypeAndState eventAndState = currentState.randomTransition(rnd);
				if (!eventAndState.state.isTerminal()) {
					// reinsert
					states.put(address, eventAndState.state);
				}

				return new Event(eventAndState.eventType, address);
			}
		}
	}