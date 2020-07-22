package com.example.newsdetector;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    public static final String REQUEST_MESSAGE = "request";
    public static final String REPLY_MESSAGE = "reply";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final EditText urlInput = findViewById(R.id.urlInput);

        final Button button = findViewById(R.id.submitButton);
        button.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                String url = urlInput.getText().toString();
                Thread tcp = new Thread(new TcpThread(url));
                tcp.start();
            }
        });
    }

    private void display(String request, String reply) {
        Intent intent = new Intent(MainActivity.this, DisplayActivity.class);
        intent.putExtra(REQUEST_MESSAGE, request);
        intent.putExtra(REPLY_MESSAGE, reply);
        startActivity(intent);
    }

    private class TcpThread implements Runnable {
        String request, reply;
        TcpThread(String request) {
            this.request = request;
        }
        @Override
        public void run() {
            try {
                Socket socket = new Socket("128.2.144.116", 5000);
                DataInputStream input = new DataInputStream(socket.getInputStream());
                DataOutputStream output = new DataOutputStream(socket.getOutputStream());
                output.write(request.getBytes());
                output.flush();
                byte[] inBytes = new byte[4096];
                input.read(inBytes);
                reply = new String(inBytes, "UTF-8");
                display(request, reply);
                input.close();
                output.close();
                socket.close();
            } catch (Exception e) {
                display(request, "C");
            }
        }
    }
}