package com.example.vlcrtux;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Handler;
import android.os.Looper;
import android.util.AttributeSet;
import android.view.View;
import android.util.Log;
import java.util.Random;


public class BrightnessOverlayView extends View {
    // Alpha value ranges from 0 to 255
    private int currentAlpha = 0;
    private boolean increasing = true;
    private Paint paint;

    public BrightnessOverlayView(Context context) {
        super(context);
        // make view transparent
        this.setBackgroundColor(Color.TRANSPARENT);
        paint = new Paint();
        startUpdateLoop();
    }

    public BrightnessOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.setBackgroundColor(Color.TRANSPARENT);
        paint = new Paint();
        startUpdateLoop();
    }

    public BrightnessOverlayView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        this.setBackgroundColor(Color.TRANSPARENT);
        paint = new Paint();
        startUpdateLoop();
    }

//    private void startUpdateLoop() {
//        final Handler handler = new Handler(Looper.getMainLooper());
//        final Runnable updateRunnable = new Runnable() {
//            @Override
//            public void run() {
//                if (increasing) {
//                    currentAlpha += 2; // Adjust this value as needed
//                    if (currentAlpha >= 255) {
//                        increasing = false;
//                    }
//                } else {
//                    currentAlpha -= 2; // Adjust this value as needed
//                    if (currentAlpha <= 0) {
//                        increasing = true;
//                    }
//                }
//                invalidate(); // Trigger a redraw of the view
//
//                // Schedule the next update
//                handler.postDelayed(this, 8); // Aim for ~120 updates per second
//            }
//        };
//        handler.post(updateRunnable);
//    }

    private void startUpdateLoop() {
        final Handler handler = new Handler(Looper.getMainLooper());
        final Runnable updateRunnable = new Runnable() {
            @Override
            public void run() {
                // Generate a random alpha value between 0 and 255
                currentAlpha = new Random().nextInt(190); // 0 (inclusive) to 256 (exclusive)

                invalidate(); // Trigger a redraw of the view
                // Schedule the next update
                handler.postDelayed(this, 8); // Aim for ~120 updates per second
            }
        };
        handler.post(updateRunnable);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        paint.setColor(Color.argb(currentAlpha, 0, 0, 0));
        canvas.drawRect(0, 0, getWidth(), getHeight(), paint);
    }
}
