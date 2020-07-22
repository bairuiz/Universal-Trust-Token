// A Java program for a Client 
import java.net.*; 
import java.io.*; 
  
public class Client 
{
    public static void main(String args[])
    {
        try {
        	Socket socket = new Socket("128.2.144.116", 5000);
            DataInputStream input = new DataInputStream(socket.getInputStream());
            DataOutputStream output = new DataOutputStream(socket.getOutputStream());
            //convert string to bytes for sending to server
            String request = "https://www.cnn.com/travel/article/eu-borders-open-but-not-to-americans-intl/index.html";
            byte[] outBytes = request.getBytes();
            output.write(outBytes);
            output.flush();

            //receive bytes from server
            byte[] inBytes = new byte[1024];
            input.read(inBytes);

            //parse string to get information
            String reply = new String(inBytes, "UTF-8");
            System.out.println(reply);
            input.close();
            output.close();
            socket.close();
        } catch (Exception e) {
            //Ignore
        }
    }
}
