// A Java program for a Client 
import java.net.*; 
import java.io.*; 
  
public class Client 
{
    public static void main(String args[])
    {
        try {
        	Socket socket = new Socket("24.6.204.81", 5000);
            DataInputStream input = new DataInputStream(socket.getInputStream());
            DataOutputStream output = new DataOutputStream(socket.getOutputStream());
            output.writeUTF("TEST CLIENT to be replaced by mobile app");
            output.flush();
            String reply = input.readUTF();
            input.close();
            output.close();
            socket.close();
        } catch (Exception e) {
            //Ignore
        }
    }
}