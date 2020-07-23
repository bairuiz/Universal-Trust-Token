package com.example.newsdetector;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.RatingBar;


public class DisplayActivity extends AppCompatActivity {
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.display);

        Intent intent = getIntent();
        String reply = intent.getStringExtra(MainActivity.REPLY_MESSAGE);

        final TextView resultTextView = findViewById(R.id.result);
        final TextView percentTextView = findViewById(R.id.percent);
        final TextView analysisTextView = findViewById(R.id.analysis);
        final RatingBar simpleRatingBar = findViewById(R.id.simpleRatingBar);
        char status = reply.isEmpty() ? 'U' : reply.charAt(0);
        switch (status) {
            case 'O':
                percentTextView.setText(reply.substring(1, 4).trim() + "%");
                analysisTextView.setText(reply.substring(4).trim());
                float rating = Float.valueOf(reply.substring(1, 4)) / 100 * 5;
                simpleRatingBar.setRating(rating);
                //int color = mixTwoColors(Color.RED, Color.YELLOW, rating / 5);
                if (rating >= 0 && rating <= 1.7) {
                    simpleRatingBar.setProgressTintList(ColorStateList.valueOf(Color.RED));
                }
                if (rating > 1.7 && rating <= 3.3) {
                    simpleRatingBar.setProgressTintList(ColorStateList.valueOf(Color.YELLOW));
                }
                if (rating > 3.3 && rating <= 5) {
                    simpleRatingBar.setProgressTintList(ColorStateList.valueOf(Color.GREEN));
                }
                resultTextView.setText("Trust Rating");
                resultTextView.setTextColor(0xFF4CAF50);
                break;
            case 'C':
                resultTextView.setText("Connection timeout");
                resultTextView.setTextColor(0xFFE91E63);
                break;
            case 'I':
                resultTextView.setText("Invalid URL");
                resultTextView.setTextColor(0xFFE91E63);
                break;
            case 'N':
                resultTextView.setText("Invalid News URL");
                resultTextView.setTextColor(0xFFE91E63);
                break;
            case 'U':
            default:
                resultTextView.setText("Unknown ERROR");
                resultTextView.setTextColor(0xFFE91E63);
                break;
        }
    }

    @Override
    public void onBackPressed() {
        Intent intent = new Intent(DisplayActivity.this, MainActivity.class);
        finish();
        startActivity(intent);
    }

    /*private static int mixTwoColors( int color1, int color2, float amount )
    {
        final byte ALPHA_CHANNEL = 24;
        final byte RED_CHANNEL   = 16;
        final byte GREEN_CHANNEL =  8;
        final byte BLUE_CHANNEL  =  0;

        final float inverseAmount = 1.0f - amount;

        int a = ((int)(((float)(color1 >> ALPHA_CHANNEL & 0xff )*amount) +
                ((float)(color2 >> ALPHA_CHANNEL & 0xff )*inverseAmount))) & 0xff;
        int r = ((int)(((float)(color1 >> RED_CHANNEL & 0xff )*amount) +
                ((float)(color2 >> RED_CHANNEL & 0xff )*inverseAmount))) & 0xff;
        int g = ((int)(((float)(color1 >> GREEN_CHANNEL & 0xff )*amount) +
                ((float)(color2 >> GREEN_CHANNEL & 0xff )*inverseAmount))) & 0xff;
        int b = ((int)(((float)(color1 & 0xff )*amount) +
                ((float)(color2 & 0xff )*inverseAmount))) & 0xff;

        return a << ALPHA_CHANNEL | r << RED_CHANNEL | g << GREEN_CHANNEL | b << BLUE_CHANNEL;
    } */
}