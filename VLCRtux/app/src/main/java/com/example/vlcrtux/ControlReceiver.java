// package com.example.vlcrtux;

// import android.content.BroadcastReceiver;
// import android.content.Context;
// import android.content.Intent;
// import android.util.Log;

// public class ControlReceiver extends BroadcastReceiver {
//     @Override
//     public void onReceive(Context context, Intent intent) {
//         String action = intent.getAction();
//         if (action == null) return;

//         Intent serviceIntent = new Intent(context, MainActivity.class);
//         serviceIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);

//         switch (action) {
//             case "com.example.vlcrtux.action.PAUSE":
//             case "com.example.vlcrtux.action.PLAY":
//             case "com.example.vlcrtux.action.WHITE":
//             case "com.example.vlcrtux.action.STEP":
//                 serviceIntent.setAction(action);
//                 context.startActivity(serviceIntent);
//                 break;
//             default:
//                 Log.d("ControlReceiver", "Unknown action: " + action);
//         }
//     }
// }

package com.example.vlcrtux;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;
import androidx.core.content.ContextCompat;

public class ControlReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        Intent serviceIntent = new Intent(context, LaunchActivityService.class);
        serviceIntent.setAction(intent.getAction());

        // Explicitly start foreground service
        try {
            ContextCompat.startForegroundService(context, serviceIntent);
            Log.d("ControlReceiver", "Foreground service started successfully.");
        } catch (Exception e) {
            Log.e("ControlReceiver", "Failed to start foreground service", e);
        }
    }
}
