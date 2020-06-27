// A Java program for a Server 
import java.net.*; 
import java.io.*; 
  
public class Server 
{
	private static final int SERVER_PORT = 5000;
	private ServerSocket server = null;
	private Socket socket = null;
	private DataInputStream input = null;
	private DataOutputStream output = null;
	private String request, reply;
	private int count = 0;

    Server() throws Exception
    {
    	server = new ServerSocket(SERVER_PORT);
    	System.out.println("Server started");
    }
    
    private void terminate() throws Exception
    {
	    server.close();
    }
    
    private void run() {
    	while (true) {
	        try {
	        	openSocket();
	        	readRequest();
	        	process();
	        	writeReply();
	        	closeSocket();
	        } catch(Exception i) {
	            System.out.println(i);
	        }
    	}
    }
    
    private void openSocket() throws Exception {
    	System.out.printf("\nWaiting for client %d ...\n", count++);
        socket = server.accept();
        System.out.println("Client accepted");
        input = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
        output = new DataOutputStream(socket.getOutputStream());
    }
    
    private void closeSocket() throws Exception {
    	input.close();
    	output.close();
    	socket.close();
    	System.out.println("Session closed.");
    }
    
    private void readRequest() throws Exception {
        request = input.readUTF();
        System.out.println("Client Request: " + request);
    }
    
    private void writeReply() throws Exception {
        System.out.println("Server Reply: " + reply);
        output.writeUTF(reply);
        output.flush();
    }
    
    private void process() throws Exception {
    	ProcessBuilder builder = new ProcessBuilder();
    	builder.command("python", "..\\Python\\WebScrapper.py", request);
    	Process p = builder.start();
    	BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
    	StringBuilder sb = new StringBuilder();
    	String line;
    	while((line = reader.readLine()) != null) {
        	sb.append(line);
    	}
    	reply = String.format("%s%3dREPLY from windows server. Testing Analysis.\n%s", (count%2==0?"T":"F"), (50+count)%101, sb.toString());
    }
  
    public static void main(String args[]) 
    {
    	try {
            Server server = new Server();
            server.run();
            server.terminate();
    	} catch (Exception e) {
    		// Ignore
    	}
    } 
}