package com.example.newsdetector;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;

public class DisplayActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.display);

        Intent intent = getIntent();
        String url = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);

        Thread tcp = new Thread(new TcpThread(url));
        tcp.start();

        final TextView urlTextView = findViewById(R.id.url);
        urlTextView.setText(url);
    }

    private void refreshDisplay(String message) {
        final TextView resultTextView = findViewById(R.id.result);
        if (message.charAt(0) == 'F') {
            resultTextView.setText("FAKE");
            resultTextView.setTextColor(0xFFE91E63);
        } else {
            resultTextView.setText("REAL");
            resultTextView.setTextColor(0xFF1EE940);
        }
        final TextView percentTextView = findViewById(R.id.percent);
        percentTextView.setText(message.substring(1, 4).trim() + "%");
        final TextView analysisTextView = findViewById(R.id.analysis);
        analysisTextView.setText(message.substring(4));
    }

    private void errorDisplay() {
        clearDisplay();
        final TextView errorTextView = findViewById(R.id.text1);
        final TextView percentTextView = findViewById(R.id.percent);
        errorTextView.setText("ERROR");
        percentTextView.setText("Unable to connect server.\nPlease try again later.");
    }

    private void clearDisplay() {
        final TextView textViews[] = new TextView[6];
        textViews[0] = findViewById(R.id.percent);
        textViews[1] = findViewById(R.id.analysis);
        textViews[2] = findViewById(R.id.text1);
        textViews[3] = findViewById(R.id.text2);
        textViews[4] = findViewById(R.id.url);
        textViews[4] = findViewById(R.id.result);
        for(TextView textView : textViews) {
            textView.setText("");
        }
    }

    private class TcpThread implements Runnable {
        String request, reply;
        TcpThread(String request) {
            this.request = request;
        }
        @Override
        public void run() {
            try {
                Socket socket = new Socket("192.168.0.6", 5000);
                DataInputStream input = new DataInputStream(socket.getInputStream());
                DataOutputStream output = new DataOutputStream(socket.getOutputStream());
                output.writeUTF(request);
                output.flush();
                reply = input.readUTF();
                refreshDisplay(reply);
                input.close();
                output.close();
                socket.close();
            } catch (Exception e) {
                errorDisplay();
            }
        }
    }
}