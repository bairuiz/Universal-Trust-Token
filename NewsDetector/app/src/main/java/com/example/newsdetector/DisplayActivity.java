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

        final TextView detailTextView = findViewById(R.id.detail);
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
                float color = rating*(Color.RED - Color.YELLOW)/5 + Color.YELLOW;
                simpleRatingBar.setProgressTintList(ColorStateList.valueOf((int) color));
                resultTextView.setText("REAL");
                resultTextView.setTextColor(0xFF4CAF50);
                detailTextView.setText("Trust Rating");
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
                resultTextView.setText("Unknown ERROR");
                resultTextView.setTextColor(0xFFE91E63);
                break;
            default:
                break;
        }
    }
}