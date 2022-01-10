import React, { Component } from 'react';
import {Image, ActivityIndicator, Alert, Text, View, TextInput, Button, SafeAreaView  } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      data: [],
      text: "",
      isLoading: true,
      image: null
    };
  }

  async pickImage () {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    console.log(result);

    if (!result.cancelled) {
      this.setState({ image: result.uri });
    }

     // ImagePicker saves the taken photo to disk and returns a local URI to it
    let localUri = result.uri;
    let filename = localUri.split('/').pop();

    // Infer the type of the image
    let match = /\.(\w+)$/.exec(filename);
    let type = match ? `image/${match[1]}` : `image`;
    
    // Upload the image using the fetch and FormData APIs
    let formData = new FormData();
    // Assume "photo" is the name of the form field the server expects
    formData.append('photo', { uri: localUri, name: filename, type });
    console.log(formData);

    return await fetch('http://10.30.254.74:5000/download', {
      method: 'POST',
      body: formData,
      headers: {
        'content-type': 'multipart/form-data',
      },
    });

  };

  async getResult(text) {
    try {
      const response = await fetch('http://10.30.254.74:5000/predict', {
        method: 'POST',
        body: JSON.stringify({'text': text}),
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const json = await response.json();
      this.setState({ data: json.result });
      console.log(json)
    } catch (error) {
      console.log(error);
    } finally {
      this.setState({ isLoading: false });
      Alert.alert(this.state.data);
    }
  }

  async initApp() {
    try {
      const response = await fetch('http://10.30.254.74:5000/');
      const json = await response.json();
      this.setState({ isLoading: false });
      console.log(json)
    } catch (error) {
      console.log(error);
    } finally {
      this.setState({ isLoading: false });
    }
  }



  componentDidMount() {
    this.initApp();
  }

  render() {
    const { data, text, isLoading } = this.state;

    return (
      <View style={{ flex: 1, justifyContent: "center"}}>
      <View style={{ backgroundColor: "black"}}>
        <Text style={{ textAlign: 'center', marginTop:30, marginBottom:20, fontSize: 20, color: "white"}}>Fake Content Detection Tool</Text>
      </View>
      <View style={{ flex: 1, justifyContent: "center" }}>
      {isLoading ? 
      (
        
        <View >
            <ActivityIndicator size="large"/>
            <Text style={{ textAlign: 'center', marginTop: 20}}>Loading...</Text>
        </View>
      ): 
      (<View style={{ flex: 1, padding: 24 }}>
          
          <TextInput
          value={text}
          onChangeText={text => this.setState({text})}
          placeholder='text here'
          style={{ textAlignVertical: 'top',}} 
          />
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          <Button title="Pick an image from camera roll" onPress={() => this.pickImage()} />
          {this.state.image && <Image source={{ uri: this.state.image }} style={{ width: 400, height: 300 }} />}
          </View>
          <Button
          title="Predict"
          onPress={() => this.getResult(text)}
          />
      </View>
      )}
    </View>
    </View>
    );
  }
};

