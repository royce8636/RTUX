package com.example.vlcrtux;

import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.util.Log;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import org.videolan.libvlc.LibVLC;
import org.videolan.libvlc.Media;
import org.videolan.libvlc.MediaPlayer;
import org.videolan.libvlc.util.VLCVideoLayout;
import java.util.ArrayList;
import android.content.Intent;
import android.view.View;
import java.io.FileOutputStream;
import android.content.Context;
import android.content.BroadcastReceiver;
import android.content.IntentFilter;
import java.io.IOException;
import android.os.Handler;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.widget.ImageView;
import android.view.WindowManager;
import android.view.View;
import android.content.pm.ActivityInfo;
import android.graphics.Color;
import android.graphics.Canvas;
import android.graphics.Paint;


//import org.videolan.libvlc.util.ThumbnailProvider;
//thumbnails provider
//import org.videolan.libvlc.gui
import com.example.vlcrtux.BrightnessOverlayView;


public class MainActivity extends AppCompatActivity {
    private LibVLC libVLC;
    private MediaPlayer mediaPlayer;
    private VLCVideoLayout videoLayout;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static final String[] PERMISSIONS_STORAGE = {
//            Manifest.permission.READ_MEDIA_VIDEO,
//            Manifest.permission.READ_MEDIA_IMAGES,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.MANAGE_EXTERNAL_STORAGE,
//            Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED
    };
    private int repeatCount;
    private int currentRepeat = 1;
    private ControlReceiver receiver;
    private ImageView thumbnailView = null;

    public class ControlReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            // Check the action of the incoming intent
            if ("com.example.vlcrtux.action.PAUSE".equals(intent.getAction())) {
                // Pause the video
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.pause();
                }
            } else if ("com.example.vlcrtux.action.PLAY".equals(intent.getAction())) {
                // Play the video
                clearVideoPlaybackFile();
                Log.d("VideoDebug", "Play action received");
                if (mediaPlayer != null) {
                    // Hide thumbnail before starting video
                    removeThumbnailView();
                    // Start video
                    mediaPlayer.play();
                } else {
                    Log.e("VideoDebug", "MediaPlayer is null.");
                }//                if (!mediaPlayer.isPlaying()) {
//                    // Remove the ImageView from its parent if it exists
//                    if (thumbnailView != null) {
//                        ViewGroup parentView = (ViewGroup) thumbnailView.getParent();
//                        if (parentView != null) {
//                            parentView.removeView(thumbnailView);
//                            thumbnailView = null; // Ensure the reference is cleared after removal
//                            Log.d("VideoDebug", "Removed view");
//                        }
//                    }
//                }
            } else if ("com.example.vlcrtux.action.WHITE".equals(intent.getAction())) {
                showWhiteScreen();
            } else if ("com.example.vlcrtux.action.STEP".equals(intent.getAction())) {
                showStepWedge();
            }

        }
    }


    private void showWhiteScreen() {
//        videoLayout.setBackgroundColor(Color.WHITE);
        Bitmap whiteImage = Bitmap.createBitmap(videoLayout.getWidth(), videoLayout.getHeight(), Bitmap.Config.ARGB_8888);
        whiteImage.eraseColor(Color.WHITE);
        if (thumbnailView == null) {
            thumbnailView = new ImageView(this);
            FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
            );
            thumbnailView.setLayoutParams(layoutParams);
            thumbnailView.setScaleType(ImageView.ScaleType.CENTER_CROP);
            ((ViewGroup) videoLayout.getParent()).addView(thumbnailView);
        }

        thumbnailView.setImageBitmap(whiteImage);
        thumbnailView.setVisibility(View.VISIBLE);
    }

    private Bitmap createStepWedge(int width, int height, int numSteps) {
        Bitmap stepWedge = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(stepWedge);

        int stepHeight = height / numSteps;
        int colorStep = 255 / (numSteps - 1);

        for (int i = 0; i < numSteps; i++) {
            int gray = i * colorStep;
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(gray, gray, gray));
            canvas.drawRect(0, i * stepHeight, width, (i + 1) * stepHeight, paint);
        }

        return stepWedge;
    }

    private void showStepWedge() {
        int width = videoLayout.getWidth();
        int height = videoLayout.getHeight();
        Bitmap stepWedgeBitmap = createStepWedge(width, height, 10);

        if (thumbnailView == null) {
            thumbnailView = new ImageView(this);
            FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
            );
            thumbnailView.setLayoutParams(layoutParams);
            thumbnailView.setScaleType(ImageView.ScaleType.CENTER_CROP);
            ((ViewGroup) videoLayout.getParent()).addView(thumbnailView);
        }

        thumbnailView.setImageBitmap(stepWedgeBitmap);
        thumbnailView.setVisibility(View.VISIBLE);
    }


    @RequiresApi(api = Build.VERSION_CODES.TIRAMISU)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
//        PowerManager.WakeLock wl = pm.newWakeLock(PowerManager.SCREEN

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        hideSystemUI();
        clearVideoPlaybackFile();
        verifyStoragePermissions();
        try {
            setupMediaPlayer();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Register the receiver
        IntentFilter filter = new IntentFilter();
        filter.addAction("com.example.vlcrtux.action.PAUSE");
        filter.addAction("com.example.vlcrtux.action.PLAY");
        filter.addAction("com.example.vlcrtux.action.WHITE");
        filter.addAction("com.example.vlcrtux.action.STEP");

//        receiver = new ControlReceiver();
//        registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED);

    }

    private void clearVideoPlaybackFile() {
        String filename = "videoplayback.txt";
        FileOutputStream outputStream = null;
        try {
            // Open the file in MODE_PRIVATE mode to overwrite its content with an empty string
            outputStream = openFileOutput(filename, MODE_PRIVATE);
            outputStream.write("".getBytes());
            Log.d("VideoDebug", "File contents cleared successfully");
        } catch (Exception e) {
            Log.e("VideoDebug", "Failed to clear file contents: " + e.toString());
        } finally {
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e("VideoDebug", "Failed to close file output stream: " + e.toString());

                }
            }
        }
    }


    private void verifyStoragePermissions() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.d("VideoDebug", "Requesting storage permission");
            ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
            Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
            startActivityForResult(intent, 999);
        } else {
            Log.d("VideoDebug", "Storage permission already granted");
        }
//        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.MANAGE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
//            Log.d("VideoDebug", "Requesting storage permission");
//            Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
//            startActivityForResult(intent, 999);
//        } else {
//            Log.d("VideoDebug", "Storage permission already granted");
//        }
    }

    private void notifyVideoStopped(String fileContents) {
        String filename = "videoplayback.txt";
//        String fileContents = "Video playback has stopped. Current repeat: " + currentRepeat + "\n";

        FileOutputStream outputStream;

        try {
            outputStream = openFileOutput(filename, MODE_PRIVATE);
            outputStream.write(fileContents.getBytes());
            outputStream.close();
            Log.d("VideoDebug", "File write successful");
        } catch (Exception e) {
//            e.printStackTrace();
            Log.e("VideoDebug", "File write failed: " + e.toString());
        }
    }


    private void setupMediaPlayer() throws IOException {
        videoLayout = findViewById(R.id.videoLayout);
        repeatCount = getIntent().getIntExtra("repeat", 1);
        Log.d("VideoDebug", "Repeat count: " + repeatCount);

        initLibVLCAndMediaPlayer();
        playVideoFromUri(getIntent().getData());
    }

    private void initLibVLCAndMediaPlayer() {
        libVLC = new LibVLC(this, new ArrayList<>());
        mediaPlayer = new MediaPlayer(libVLC);
        mediaPlayer.attachViews(videoLayout, null, false, false);
        mediaPlayer.setEventListener(this::handleMediaPlayerEvents);
    }

    private Bitmap getFirstFrame(Uri videoUri) throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(this, videoUri);

        Bitmap firstFrame = null;
        int retryCount = 0;
        final int maxRetries = 20;

        while (firstFrame == null && retryCount < maxRetries) {
            try {
                firstFrame = retriever.getFrameAtTime(100, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                if (firstFrame != null) {
                    Log.d("VideoDebug", "First frame retrieved successfully");
                }
            } catch (IllegalArgumentException e) {
                Log.e("VideoDebug", "IllegalArgumentException: " + e.getMessage());
                break;
            } catch (RuntimeException e) {
                Log.e("VideoDebug", "RuntimeException: " + e.getMessage());
                break;
            }

            if (firstFrame == null) {
                try {
                    Log.d("VideoDebug", "Retrying frame retrieval");
                    Thread.sleep(10);  // Short delay between retries (adjust as needed)
                    retryCount++;
                } catch (InterruptedException e) {
                    Log.w("VideoDebug", "Interrupted during frame retrieval wait: " + e.getMessage());
                    break;
                }
            }
        }
        retriever.release();
        return firstFrame;
    }

    private void playVideoFromUri(Uri videoUri) throws IOException {
        if (videoUri == null) {
            Log.e("VideoDebug", "Video URI is null");
            return;
        }

//        Bitmap firstFrame = getFirstFrame(videoUri);
        String videoPath = videoUri.getPath();
        assert videoPath != null;
        String thumbPath = videoPath.replace("vid.avi", "thumbnail.bmp");
        Log.d("VideoDebug", "Thumbnail path: " + thumbPath);
        Bitmap firstFrame = BitmapFactory.decodeFile(thumbPath);
        if (firstFrame != null) {
            if (firstFrame.getWidth() > firstFrame.getHeight()) {
                setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE);
            } else {
                setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT);
            }
            BitmapDrawable drawable = new BitmapDrawable(getResources(), firstFrame);
            // drawable.setBounds(0, 0, firstFrame.getWidth(), firstFrame.getHeight());
            thumbnailView = new ImageView(this);
            thumbnailView.setImageBitmap(firstFrame);

            FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, 
                FrameLayout.LayoutParams.MATCH_PARENT
            );
            thumbnailView.setLayoutParams(layoutParams);
            thumbnailView.setScaleType(ImageView.ScaleType.CENTER_CROP); // or use FIT_XY for exact match

            ((ViewGroup) videoLayout.getParent()).addView(thumbnailView, 0); // Add to the front
            thumbnailView.bringToFront();
            thumbnailView.setVisibility(View.VISIBLE);
            Log.d("VideoDebug", "First frame set as background");
            notifyVideoStopped("ready\n");
        } else {
            Log.e("VideoDebug", "Failed to get the first frame of the video.");
        }

        Media media = new Media(libVLC, videoUri);
        mediaPlayer.setMedia(media);
    }


    private void handleMediaPlayerEvents(MediaPlayer.Event event) {
        switch (event.type){
            case MediaPlayer.Event.Playing:
                removeThumbnailView();
                break;
            case MediaPlayer.Event.EndReached:
                notifyVideoStopped("stopped\n");
                Log.d("VideoDebug", "Video ended. Repeat count: " + currentRepeat);
                mediaPlayer.pause();  // Pause before releasing resources

                // Introduce a slight delay to show the last frame
                new Handler().postDelayed(() -> {
                    mediaPlayer.stop();
                    currentRepeat++;
                    hideSystemUI();
                }, 1);
        }
    }


    private void removeThumbnailView() {
        runOnUiThread(() -> {
            if (thumbnailView != null) {
                ViewGroup parentView = (ViewGroup) thumbnailView.getParent();
                if (parentView != null) {
                    parentView.removeView(thumbnailView);
                    thumbnailView = null; // Ensure the reference is cleared after removal
                    Log.d("VideoDebug", "Removed thumbnail view");
                }
            }
        });
    }

    private void handleNewVideo(Intent intent) throws IOException {
        Uri videoUri = intent.getData();
        repeatCount = intent.getIntExtra("repeat", 1);
        currentRepeat = 1; // Reset the current repeat count

        if (videoUri != null) {
            // Consider resetting or reinitializing mediaPlayer here if needed
            playVideoFromUri(videoUri);
        } else {
            Log.e("VideoDebug", "Video URI is null in new intent");
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);

        if (intent.getAction() != null) {
            switch (intent.getAction()) {
                case "com.example.vlcrtux.action.PAUSE":
                    if (mediaPlayer.isPlaying()) mediaPlayer.pause();
                    break;
                case "com.example.vlcrtux.action.PLAY":
                    if (!mediaPlayer.isPlaying()) mediaPlayer.play();
                    break;
                case "com.example.vlcrtux.action.WHITE":
                    showWhiteScreen();
                    break;
                case "com.example.vlcrtux.action.STEP":
                    showStepWedge();
                    break;
            }
        }
    }


    //    @Override
//    protected void onNewIntent(Intent intent) {
//        super.onNewIntent(intent);
//        setIntent(intent); // Update the intent
//        // Check if file is given
//        if (intent.getData() != null) {
//            Log.d("VideoDebug", "New video received");
//            clearVideoPlaybackFile();
//            try {
//                handleNewVideo(intent);
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//        }
//
//    }

    private void hideSystemUI() {
        // Enables regular immersive mode.
        // For "lean back" mode, remove SYSTEM_UI_FLAG_IMMERSIVE.
        // Or for "sticky immersive," replace it with SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        View decorView = getWindow().getDecorView();
        decorView.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        // Hide the nav bar and status bar
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            hideSystemUI();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mediaPlayer.release();
        libVLC.release();
        clearVideoPlaybackFile();
        unregisterReceiver(receiver);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_EXTERNAL_STORAGE && grantResults.length > 0) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d("VideoDebug", "Storage permission granted");
            } else {
                Log.e("VideoDebug", "Storage permission denied");
            }
        }
    }
}
