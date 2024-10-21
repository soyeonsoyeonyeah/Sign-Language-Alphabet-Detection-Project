package spring.lstm.controller;

import java.nio.charset.Charset;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

@Controller
public class LstmController {
	// webcam.html 이동 함수 
	@RequestMapping(value="/webcamControl.do", method=RequestMethod.GET)
	public ModelAndView showWebCam() {
		ModelAndView mav = new ModelAndView();
		mav.setViewName("webcam.html");
		return mav;
	}
	
	// RestServer로 이미지 데이터 보내는 함수
	@ResponseBody // Html 같은 화면으로 이동하지 않고 결과를 리턴
	@RequestMapping(value="/sendLstm.do", method=RequestMethod.POST)
	public String sendImageToRestServer(@RequestBody String data) throws Exception {
		// 웹캠 이미지 데이터
		data = data.replace("data:", "");
		System.out.println("data(after replace)=" + data);
		
		//데이터 Json 객체로 변환
		JSONObject dataObject = new JSONObject(data);
		System.out.println("dataObject=" + dataObject);
		
		// 위 데이터에서 이미지 데이터만 뽑아냄
		JSONArray cam_data_arr = ((JSONArray)dataObject.get("img_data"));
		System.out.println("cam_data_arr=" + cam_data_arr);
		
		// RestServer로 전송할 Json 객체 생성
		JSONObject restSendData = new JSONObject();
		restSendData.put("data", cam_data_arr);
		System.out.println("restSendData=" + restSendData);
		
		// 접속할 RestServer Url
		HttpPost httpPost = new HttpPost("http://localhost:5000/lstm_detect");
		// 전송할 데이터 타입 설정
		httpPost.addHeader("Content-Type", "application/json;charset=utf-8");
		// 리턴 받을 데이터 타입 설정
		httpPost.setHeader("Accept", "application/json;charset=utf-8");
		
		// 캠 화면 이미지 저장
		StringEntity stirngEntity = new StringEntity(restSendData.toString());
		// Rest Server로 전송할 객체 생성
		httpPost.setEntity(stirngEntity);
		
		// RestServer의 함수를 호출할 객체 생성
		CloseableHttpClient httpClient = HttpClients.createDefault();
		// Rest 서버의 함수를 호출하고 리턴값을 가져올 객체 생성
		CloseableHttpResponse response2 = httpClient.execute(httpPost);
		
		// Rest서버의 리턴값을 string으로 변환하여 저장
		String lstm_message = EntityUtils.toString(response2.getEntity(), Charset.forName("UTF-8"));
		
		return lstm_message;
	}
}
