// A Java program for a Server 
import java.net.*; 
import java.io.*; 
  
public class Server 
{
	ServerSocket server = null;

    public Server(int port)
    {
    	try {
	        server = new ServerSocket(port);
	        System.out.println("Server started");
    	} catch (Exception e) {
    		//
    	}
    }
    
    public void close()
    {
    	try {
	        server.close();
    	} catch (Exception e) {
    		//
    	}
    }
    
    public void listen() {
    	int count = 0;
    	while (count < 15) {
	        try {
	        	System.out.printf("\nWaiting for client %d ...\n", count++);
	            Socket socket = server.accept();
	            System.out.println("Client accepted");
	  
	            // takes input from the client socket
	            DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
	            String request = in.readUTF();
	            System.out.println("Client Request: " + request);
	            
	            // writes output to client socket
	            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
	            String garbage = "sdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsnsdfnkjabsjdfoisndfsn";
	            String reply = String.format("%s%3dREPLY from windows server. Testing Analysis.%s", (count%2==0?"T":"F"), 80+count, garbage);
	            System.out.println("Server Reply: " + reply);
	            out.writeUTF(reply);
	  
	            // close connection
	            System.out.println("Closing connection");
	            socket.close();
	            in.close();
	            out.close();
	        } catch(Exception i) {
	            System.out.println(i);
	        }
    	}
    }
  
    public static void main(String args[]) 
    { 
        Server server = new Server(5000);
        server.listen();
        server.close();
    } 
}