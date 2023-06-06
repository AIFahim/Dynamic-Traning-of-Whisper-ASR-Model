# Dynamic-Traning-of-Whisper-ASR-Model
This code 900hr of audio data (approx. 693k+ files) training procedure. So that if we also add augmentations on it, then the amount of dataset become
twitch the 693K+ files.
To do this, we faced:
- OS Memory out of bound Error after 25% of original dataset processed. That time add augmentation was became impossible for us because you couldn’t performed only processing in full original dataset.

So that, With dynamic training approach, we always storing processed & augmented datas in the virtual container. Then in our training it always feed into the model with the batch size amount of data from the virtual container.

## What is Dynamic Tranining
Dynamic Training is virtual container based Training Procedure where we processed data and performed augmentations in prior and keep storing these augmentations to the virtual container then perform training. In the Dynamic Training is virtual container based Training Procedure
where we processed data and performed augmentations in prior and keep storing these augmentations to the virtual container then perform training.
![Screenshot 2023-06-06 114337](https://github.com/AIFahim/Dynamic-Traning-of-Whisper-ASR-Model/assets/33654834/5c6023dc-75ee-4b34-baee-8d7a49f5db06)
