# Car Racing With AI!!

Yeah so basically this was my attempt for player2 ysws by hackclub! As visible we use a neat implementation over a custom car racing game where you play against the ai and get absolutely demolished.

This is my first repo like ever so sorry for any obvious mistakes!

A little bit of AI ie LLms was used for the ui and some fixes to show. 

#### The Current Game includes:
- Normal Steering and Car Mechanics
- Friction + Drifting :)
- We have a randomly generated track to add variability in the game for ai training.
- We use Catmull Rom interpolation for smoothing a set of 24 points we randomly pick on an elliptical path with some variation
- Integration of NEAT neuro evolution for the ai part

#### How to Play? -
- Clone the repository
- run the command -> "python main.py --replay --fullscreen" to play against the ai Realtime while viewing its decision in the neural network
- To train the ai, just run python main.py --debug to view training live, Modify the Render var in code to TRUE to actually see the train in game (debug only does not render only prints telemetry)

#### An image of the Game

<img width="1526" height="858" alt="image" src="https://github.com/user-attachments/assets/e7c99674-ea90-4edd-824a-0e321212ccd8" />
Ai so fast that even I couldnt click a photo of it 

### Happy Racing 🏎️
