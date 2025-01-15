package parser

import (
	"encoding/json"
	"fmt"
	"os"
)

func CreateJsonFileFromStruct(data any, pathToSave string) error {
	fileData, err := json.MarshalIndent(data, "", "	")
	if err != nil {
		return fmt.Errorf("failed to marshal data. Error: %w\n", err)
	}

	file, err := os.Create(pathToSave)
	if err != nil {
		return fmt.Errorf("failed to create file. Error: %w\n", err)
	}
	defer file.Close()

	_, err = file.Write(fileData)
	if err != nil {
		return fmt.Errorf("failed to write data to file. Error: %w\n", err)
	}

	return nil
}
