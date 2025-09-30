The stages of this project:

- [x] Create minimum project requirements
- [ ] Create plan for achieving project requirements

Deliverables:

- [ ] The creation of a wireless remote containing two movable arms that can be used to control the joint angles on a hexapod robot.
  - [ ] Arms Section
    - [ ] Figure out how to modify MG90s servos to get position feedback
      - [ ] Could just tap into the potentiometer reading and break that out?
      - [ ] Alternatively design a new board that can be fit into the housing of a servo with minimal changes that allows for position feedback and current sensing (optional).
      - [ ] Current sensing on the power pins to the servos, with a power MOSFET to switch (i.e. so we can do 'current' control)
    - [ ] Figure out mechanics for dummy arms on controller
  - [ ] The Remote Section
    - [ ] Figure out wireless data transfer between remote and raspberry pi
      - [ ] Could use NRF24L01s, with the receiver being a USB dongle that plugs into the rasperry pi
        - [ ] This would need custom drivers probably, but could be fun
      - [ ] Some other method, BT, BLE, etc
    - [ ] Creation of a PCB that houses all of these elements
    - [ ] Joysticks
    - [ ] Buttons
    - [ ] Sliders / Faders ?
    - [ ] Debug ports, test pads / pins
    - [ ] The two arms connection