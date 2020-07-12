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
        String request = intent.getStringExtra(MainActivity.REQUEST_MESSAGE);
        String reply = intent.getStringExtra(MainActivity.REPLY_MESSAGE);

        final TextView urlTextView = findViewById(R.id.url);
        final TextView accuracyTextView = findViewById(R.id.accuracy);
        final TextView resultTextView = findViewById(R.id.result);
        final TextView percentTextView = findViewById(R.id.percent);
        final TextView analysisTextView = findViewById(R.id.analysis);
        final RatingBar simpleRatingBar = findViewById(R.id.simpleRatingBar);
        urlTextView.setText(request);
        if (reply.equals("ERROR")) {
            resultTextView.setText("ERROR");
            resultTextView.setTextColor(0xFFE91E63);
            accuracyTextView.setText("Please try again later.");
            return;
        } else if (reply.charAt(0) == 'F') {
            resultTextView.setText("FAKE");
            resultTextView.setTextColor(0xFFE91E63);
        } else {
            resultTextView.setText("REAL");
            resultTextView.setTextColor(0xFF1EE940);
        }
        percentTextView.setText(reply.substring(1, 4).trim() + "%");
        analysisTextView.setText(reply.substring(4).trim());
        float rating = Float.valueOf(reply.substring(1, 4)) / 100 * 5;
        simpleRatingBar.setRating(rating);
        float color = rating*(Color.RED - Color.YELLOW)/5 + Color.YELLOW;
        simpleRatingBar.setProgressTintList(ColorStateList.valueOf((int) color));
    }
}