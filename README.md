# Distributed Diffusion
Train Stable Diffusion models across the internet with multiple peers

## Contributing & Support

All contributions are welcome! Currently I'm the only developer in this project so I would be happy for more people to join!

Soon I will set up a patreon or some way to get donations, as testing this is very expensive. 

If you need help, or want to know more about the project, join the discord: https://discord.gg/8Sh2T6gjd2

## Issues & Bugs:

- Reporting DHT maddrs from peers in a VERY incorrect manner, prevents a node from the same ip closing and reconnecting.
- Too many dead DHT maddrs from peers prevent proper initialization of the DHT client. Add a ping module from the dataset server
- Memory leak requiring to run `killall python` everytime the trainer is closed
- No security whatsoever

## Credits:
- Haru: Wrote the first trainer and rewrote it from scratch optimizing it
- dep (me): Hivemind integration, dataset server
