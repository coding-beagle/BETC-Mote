The stages of this project:

- [x] Create minimum project requirements
- [ ] Create plan for achieving project requirements

Deliverables / Proposed project scope:

- [ ] Creation of an IMU based joint angle sensor for human robot interaction
  - [ ] Milestone 1: The first node
    - A single microcontroller / imu combo that is capable of measuring rotation.
      - Potential options for this include the Seeed Xiao NRF52840 sense, which has LiPo battery charging and a microcontroller w/ BLE built in
    - This will be used to test strategies for sensor data fusion + combatting sensor drift.
    - Also be used to test feasible data acquisition + transmission rates, i.e. whether BLE will be sufficient or a wired connection will be fine.
  - [ ] Milestone 2: Prototype of whole arm
    - 4 x of a working node design to test the data pipeline of four sensors simultaneously.
    - This can be achieved before / during winter break.
  - [ ] Milestone 3: Testing the device in ROS Gazebo / other simulator
    - Set up of various simulation environments for operators to practice collecting samples
    - Test simulation could also include path tracing exercises
    - We will compare the approach of a conventional joystick, this device with only end effector position mapping, and full fusion joint mapping.

- [ ] Do Academ Malprac Aware.

