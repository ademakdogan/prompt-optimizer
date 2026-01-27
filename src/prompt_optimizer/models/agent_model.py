"""
User-defined extraction schema for the Agent model.

This module provides the Pydantic model that defines what fields
the Agent should extract from input data. Users should customize
this model based on their specific extraction needs.

Example use cases:
- Personal information extraction
- Resume parsing
- Invoice data extraction
- Document field extraction
"""

from typing import Optional, Type

from pydantic import BaseModel, Field


class ExtractionSchema(BaseModel):
    """
    Schema defining the structured data to extract from input text.

    This is the output model for the Agent. Users should customize
    these fields based on their specific extraction needs.

    The fields below are examples for personal information extraction.
    Modify as needed for resumes, invoices, or other data types.

    Attributes:
        firstname: The first name of the person.
        lastname: The last name of the person.
        prefix: Honorifics or titles like Mr., Mrs., Dr.
        age: Age of the person.
        eyecolor: Eye color description.
        jobtitle: Job title or professional role.
        email: Email address found in the text.
        phonenumber: Phone number contact.
        street: Street address or physical location.
        county: County or region name.
        accountnumber: Bank or service account number.
        amount: Monetary amount including currency symbol if present.
        currency: Currency code like USD, EUR.
        maskednumber: Partial or masked card/account numbers.
        pin: Personal Identification Number or security code.

    Examples:
        >>> schema = ExtractionSchema(
        ...     firstname="John",
        ...     email="john@example.com",
        ...     amount="$1,250.00"
        ... )
        >>> schema.firstname
        'John'
    """

    # Personal Information
    firstname: Optional[str] = Field(
        None, description="The first name of the person."
    )
    lastname: Optional[str] = Field(
        None, description="The last name of the person."
    )
    prefix: Optional[str] = Field(
        None, description="Honorifics or titles like Mr., Mrs., Dr."
    )
    age: Optional[str] = Field(
        None, description="Age of the person."
    )
    eyecolor: Optional[str] = Field(
        None, description="Eye color description."
    )
    jobtitle: Optional[str] = Field(
        None, description="Job title or professional role."
    )

    # Contact Information
    email: Optional[str] = Field(
        None, description="Email address found in the text."
    )
    phonenumber: Optional[str] = Field(
        None, description="Phone number contact."
    )
    street: Optional[str] = Field(
        None, description="Street address or physical location."
    )
    county: Optional[str] = Field(
        None, description="County or region name."
    )

    # Financial Information
    accountnumber: Optional[str] = Field(
        None, description="Bank or service account number."
    )
    amount: Optional[str] = Field(
        None,
        description="Monetary amount including currency symbol if present.",
    )
    currency: Optional[str] = Field(
        None, description="Currency code like USD, EUR."
    )
    maskednumber: Optional[str] = Field(
        None,
        description="Partial or masked card/account numbers (e.g., last 4 digits).",
    )
    pin: Optional[str] = Field(
        None, description="Personal Identification Number or security code."
    )


def generate_default_prompt(schema: Type[BaseModel] = ExtractionSchema) -> str:
    """
    Generate a default extraction prompt from the schema fields.

    This function dynamically builds a prompt at runtime based on the
    fields defined in the ExtractionSchema. When the schema changes,
    the generated prompt will automatically reflect those changes.

    Args:
        schema: The Pydantic model class to generate prompt from.
                Defaults to ExtractionSchema.

    Returns:
        str: A formatted prompt string with field names and descriptions.

    Examples:
        >>> prompt = generate_default_prompt()
        >>> "Extract the following information" in prompt
        True
        >>> "firstname" in prompt
        True
    """
    # Get schema properties from the model
    model_schema = schema.model_json_schema()
    properties = model_schema.get("properties", {})

    # Build field list with descriptions
    field_lines = []
    for field_name, field_info in properties.items():
        description = field_info.get("description", "")
        if description:
            field_lines.append(f"- {field_name}: {description}")
        else:
            field_lines.append(f"- {field_name}")

    fields_text = "\n".join(field_lines)

    prompt = f"""Extract the following information from the data:

{fields_text}

Return the extracted information as a JSON object. Only include fields that are found in the text. Omit fields that are not present."""

    return prompt


def get_schema_field_descriptions(
    schema: Type[BaseModel] = ExtractionSchema,
) -> dict[str, str]:
    """
    Get field descriptions from the schema as a dictionary.

    Args:
        schema: The Pydantic model class to get descriptions from.

    Returns:
        dict[str, str]: Mapping of field names to their descriptions.

    Examples:
        >>> descriptions = get_schema_field_descriptions()
        >>> "firstname" in descriptions
        True
    """
    model_schema = schema.model_json_schema()
    properties = model_schema.get("properties", {})

    descriptions = {}
    for field_name, field_info in properties.items():
        descriptions[field_name] = field_info.get("description", "")

    return descriptions

