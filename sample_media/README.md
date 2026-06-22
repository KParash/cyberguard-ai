# Sample Media for Testing

## Where to Get Test Images/Video

### Deepfake Test Images
- **Real faces**: Use any portrait photo from [Unsplash](https://unsplash.com/s/photos/portrait) or [ThisPersonDoesExist.com](https://thispersondoesnotexist.com/)
- **Deepfake faces**: Download samples from [FaceForensics++](https://github.com/ondyari/FaceForensics) or generate with publicly available deepfake tools
- **Quick test**: Take a selfie (should classify as "Real") vs. a screenshot of an AI-generated face from Midjourney/DALL-E (may classify as "Fake")

### Video Test Files
- Any short `.mp4` video of a face will work — the system extracts 5 evenly-spaced frames and runs each through the image classifier
- Recommended: Keep videos under 30 seconds for fast processing during demos
- Sources: Record a short webcam clip, or download sample video calls from YouTube

### Notes
- The deepfake model (`prithivMLmods/Deep-Fake-Detector-v2-Model`) is a ViT-base model fine-tuned on face images at 224×224 resolution
- It works best on **close-up face shots** — full-body or group photos may give less reliable results
- No audio analysis is performed — this is image/frame-level detection only
