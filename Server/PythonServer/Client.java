// A Java program for a Client 
import java.net.*; 
import java.io.*; 
  
public class Client 
{
    public static void main(String args[])
    {
        try {
        	Socket socket = new Socket("localhost", 5000);
            System.out.println("Client running");
            DataInputStream input = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            DataOutputStream output = new DataOutputStream(socket.getOutputStream());

            //To be replaced with actual request from app
            String request = "https://www.cnn.com/travel/article/eu-borders-open-but-not-to-americans-intl/index.html";

            //convert string to bytes for sending to server
            byte[] outBytes = request.getBytes();
            output.write(outBytes);
            output.flush();

            //receive bytes from server
            byte[] inBytes = new byte[1024];
            input.read(inBytes);

            //parse string to get information
            String reply = new String(inBytes, "UTF-8");
            String title = reply.split("title:")[1].split(", percentage:")[0].trim();
            String percentage = reply.split("percentage:")[1].trim();
            System.out.println("The title of the article is " + title);
            System.out.println("This article is " + percentage + "% real");
            input.close();
            output.close();
            socket.close();
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}