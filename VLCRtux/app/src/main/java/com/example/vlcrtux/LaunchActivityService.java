package com.example.vlcrtux;

import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import androidx.core.app.NotificationCompat;

public class LaunchActivityService extends Service {

    private static final String CHANNEL_ID = "vlcrtux_channel_id";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent.getAction();

        // Intent to launch your MainActivity
        Intent activityIntent = new Intent(this, MainActivity.class);
        activityIntent.setAction(action);
        activityIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);

        PendingIntent pendingIntent = PendingIntent.getActivity(
            this, 0, activityIntent,
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ?
                PendingIntent.FLAG_IMMUTABLE : PendingIntent.FLAG_UPDATE_CURRENT);

        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Launching VLC Rtux")
            .setContentText("Starting video playback...")
            .setSmallIcon(R.drawable.ic_launcher_foreground) // Use your own icon
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build();

        startForeground(1, notification);

        // Start activity explicitly
        startActivity(activityIntent);

        // Immediately stop service after launching activity
        stopForeground(true);
        stopSelf();

        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
